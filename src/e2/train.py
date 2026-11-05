"""One explicitly authorized official E2 run; no job submission or test evaluator."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import datetime as dt
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import atomic_json, atomic_text, read_checkpoint, restore_checkpoint, save_checkpoint
from .config import (CONTRACT_SHA256, ROOT, fingerprint, read_json, require_approval,
                     sha256_file, validate_run, topology)
from .data import FixedManifest, make_loaders
from .engine import UpdateScheduler, evaluate, select_checkpoint, train_epoch
from .model import build_e2, build_optimizer


from .runtime import inspect_target_inputs, prepare as _setup
from .config import scientific_identity


def approved_identity(config: dict) -> dict:
    config = validate_run(config)
    paths = config['paths']
    acceptance_path = Path(paths['acceptance_record'])
    acceptance = require_approval(acceptance_path, 'E2-G2')
    evidence_path = Path(acceptance['acceptance_evidence'])
    if sha256_file(evidence_path) != acceptance['acceptance_evidence_sha256']:
        raise ValueError('Accepted E2 target evidence hash mismatch')
    evidence = read_json(evidence_path)
    if (evidence.get('status') != 'pass' or evidence.get('kind') != 'e2_target_acceptance_v1'
            or evidence.get('experiment') != 'E2' or evidence.get('views') != config['views']
            or evidence.get('contract_sha256') != CONTRACT_SHA256 or evidence.get('official_run') is not False
            or evidence.get('test_images_opened') != 0
            or scientific_identity(evidence.get('identity')) != scientific_identity(acceptance['identity'])):
        raise ValueError('Passing E2-specific target evidence with accepted identity required')
    gate = 'E2-G3a' if config['stage'] == 'tuning' else 'E2-G3b'
    approval = require_approval(Path(paths['authorization_record']), gate, config['run_id'])
    if approval.get('acceptance_sha256') != sha256_file(acceptance_path):
        raise ValueError('Stage receipt must bind the exact E2-G2 receipt')
    expected_ids = ({f"e2-tuning-lr{lr:.8f}-seed1001" for lr in config['recipe']['campaign']['lr_candidates']}
                    if config['stage'] == 'tuning' else
                    {f"e2-final-lr{config['lr']:.8f}-seed{seed}" for seed in (2001,3001,4001)})
    if len(approval.get('run_ids',[])) != len(expected_ids) or set(approval['run_ids']) != expected_ids:
        raise ValueError('Stage receipt must authorize exactly the reviewed four tuning or three final runs')
    declarations = approval.get('declarations',{})
    if (set(declarations) != expected_ids or declarations.get(config['run_id']) != config['config_sha256']
            or any(not isinstance(v,str) or len(v) != 64 for v in declarations.values())):
        raise ValueError('Stage receipt must bind every exact declaration digest')
    if config['stage'] == 'final':
        from .campaign import read_selection
        selection = read_selection(Path(paths['selection']), config['recipe'])
        if selection['lr'] != config['lr'] or approval.get('selection_sha256') != config['selection_sha256']:
            raise ValueError('E2-G3b does not bind this locked final selection')
    return acceptance['identity']


def preflight(config: dict) -> dict:
    config = validate_run(config)
    accepted = approved_identity(config)
    checked = inspect_target_inputs(config)
    if scientific_identity(checked) != scientific_identity(accepted):
        raise ValueError('Consumed-source/runtime/data changed; renewed E2 target acceptance required')
    return checked


@contextlib.contextmanager
def run_lock(folder: Path):
    with (Path(folder) / ".run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _event(folder, event, **extra):
    path = folder / "events.json"
    events = read_json(path) if path.exists() else []
    events.append({"event": event, "utc": dt.datetime.now(dt.timezone.utc).isoformat(), **extra})
    atomic_json(path, events)


def write_history(folder, history):
    atomic_json(folder / "history.json", history)
    output = io.StringIO()
    fields = ["epoch", "train_loss", "validation_nll", "top1", "top3", "top5", "macro_f1_supported", "weighted_f1", "optimizer_updates", "lr_first", "lr_last", "epoch_seconds"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in history:
        writer.writerow({"epoch": row["epoch"], "train_loss": row["train"]["loss"], "validation_nll": row["validation"]["nll"],
                         **{k: row["validation"][k] for k in fields[3:8]}, **{k: row["train"][k] for k in fields[8:11]}, "epoch_seconds": row["epoch_seconds"]})
    atomic_text(folder / "metrics.csv", output.getvalue())


def _checkpoint_args(context):
    return {k: context[k] for k in ("model", "optimizer", "scheduler", "scaler", "config", "identity", "loaders")}


def _attempt_snapshot(path, value):
    """Publish a complete immutable record atomically, without replacing one."""
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and exclusive on the target Linux/NTFS
        # filesystems. Unlike replace(), it can never overwrite audit evidence.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_attempt_metadata(path):
    """Return a parsed object or an exact-byte issue for an interrupted record."""
    path = Path(path)
    if not path.exists():
        return None, {'issue': 'missing'}
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('Attempt metadata must be an object')
    except (ValueError, UnicodeError):
        return None, {'issue': 'invalid_json_object', 'bytes': len(raw), 'sha256': sha256_file(path)}
    return value, None


def _begin_attempt(folder, config, operation):
    root = folder / "attempts"
    root.mkdir(exist_ok=True)
    previous = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    for old in previous:
        outcome, issue = read_attempt_metadata(old / 'outcome.json')
        if issue is not None and not (old / 'abandoned.json').exists():
            metadata = {name: read_attempt_metadata(old / name) for name in
                        ('request.json', 'environment.json', 'start_boundary.json', 'outcome.json')}
            observation = {"status": "terminated_without_outcome", "uncommitted_work": "unknown after abrupt termination",
                           **{k: config[k] for k in ('run_id', 'config_sha256', 'contract_sha256')},
                           'metadata_issues': {name: result[1] for name, result in metadata.items() if result[1] is not None},
                           "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            _attempt_snapshot(old / "abandoned.json", observation)
        if (old / 'abandoned.json').exists():
            observation = read_json(old / 'abandoned.json')
            events = read_json(folder / 'events.json') if (folder / 'events.json').exists() else []
            # Reconcile a kill between immutable publication and its event, but
            # never rewrite an existing observation or the interrupted raw bytes.
            environment, env_issue = read_attempt_metadata(old / 'environment.json')
            rel = f'attempts/{old.name}/environment.json'
            if env_issue is None and not any(e.get('environment_path') == rel for e in events):
                _event(folder, 'attempt_metadata_reconciled', attempt_id=old.name,
                       environment_path=rel, environment_sha256=sha256_file(old / 'environment.json'))
            if any(e.get('event') == 'attempt_abandoned' and e.get('attempt_id') == old.name for e in events):
                continue
            _event(folder, "attempt_abandoned", attempt_id=old.name, **observation)
    number = max((int(p.name) for p in previous), default=0) + 1
    attempt = root / f"{number:06d}"
    attempt.mkdir()
    _attempt_snapshot(attempt / "request.json", {"run_id": config["run_id"], "config_sha256": config["config_sha256"],
                      "contract_sha256": CONTRACT_SHA256, "operation": operation,
                      "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    return attempt


def _last_boundary(folder, context):
    if not (folder / "last.pt").exists():
        return {"completed_epoch": 0, "updates": 0}
    if context is not None:
        try:
            state = read_checkpoint(folder / "last.pt", config=context["config"], identity=context["identity"])
            return {k: state["progress"][k] for k in ("completed_epoch", "updates")}
        except Exception:
            # This is only a diagnostic probe in the failure handler. Preserve the
            # original load/validation error rather than failing a second time here.
            pass
    return {"completed_epoch": None, "updates": None, "state": "unverified_before_preflight_or_unreadable"}


def run(config: dict, run_dir: Path, *, resume: Path | None = None, stop_requested=lambda: False) -> dict:
    config = validate_run(config)
    folder = Path(run_dir).resolve()
    if folder != (Path(config["paths"]["runs_dir"]) / config["run_id"]).resolve():
        raise ValueError("Run directory must match its immutable declaration")
    recovering_initialization = False
    if resume is None:
        if folder.exists():
            if ((folder / "last.pt").exists() or not (folder / "config_resolved.json").is_file()
                    or read_json(folder / "config_resolved.json") != config):
                raise FileExistsError("Existing run requires unchanged declaration and explicit last.pt resume")
            recovering_initialization = True
        else:
            folder.mkdir(parents=True, exist_ok=False)
            atomic_json(folder / "config_resolved.json", config)
    elif (Path(resume).resolve() != folder / "last.pt" or not Path(resume).is_file()
          or read_json(folder / "config_resolved.json") != config):
        raise ValueError("Resume requires the same declared run and last.pt")
    with run_lock(folder):
        if resume is None and (folder / "last.pt").exists():
            raise FileExistsError("Run already initialized; use explicit last.pt resume")
        if resume and (folder / "run_summary.json").exists() and read_json(folder / "run_summary.json").get("status") == "complete":
            raise ValueError("Completed run cannot be extended or rerun")
        operation = "resume" if resume else "recover_initialization" if recovering_initialization else "created"
        attempt = _begin_attempt(folder, config, operation)
        attempt_started = time.perf_counter()
        context = None
        training = None
        in_flight_epoch = None
        _event(folder, operation, attempt_id=attempt.name, replay_policy="last committed epoch boundary; original budget")
        atomic_json(folder / "run_summary.json", {"status": "running", "run_id": config["run_id"], "contract_sha256": CONTRACT_SHA256})
        try:
            accepted_resume = None
            if resume:
                accepted_resume = approved_identity(config)
                state = read_checkpoint(resume, config=config, identity=accepted_resume)
                if state['role'] != 'last':
                    raise ValueError('Resume checkpoint role mismatch')
                del state
            checked = preflight(config)
            if accepted_resume is not None and scientific_identity(checked) != scientific_identity(accepted_resume):
                raise ValueError('Accepted resume identity changed during preflight')
            context = _setup(config, checked)
            cfg = context["config"]
            recipe = cfg["recipe"]
            args = _checkpoint_args(context)
            _attempt_snapshot(attempt / "environment.json", context["environment"])
            _event(folder, "attempt_environment", attempt_id=attempt.name, environment_path=f"attempts/{attempt.name}/environment.json",
                   environment_sha256=sha256_file(attempt / "environment.json"))
            # The root file is a latest-attempt convenience alias; immutable snapshots retain every allocation.
            atomic_json(folder / "environment.json", context["environment"])
            atomic_json(folder / "model_metadata.json", context["model_metadata"])
            progress = {"completed_epoch": 0, "updates": 0, "history": [], "selected": None, "selected_weights": None}
            if resume:
                progress = restore_checkpoint(resume, **args)
            else:
                save_checkpoint(folder / "last.pt", progress=progress, **args)
            boundary = {k: progress[k] for k in ("completed_epoch", "updates")}
            _attempt_snapshot(attempt / "start_boundary.json", boundary)
            _event(folder, "attempt_boundary", attempt_id=attempt.name, **boundary,
                   replay_policy="Any work after this boundary in a prior attempt is replayed; unknown work remains explicitly unknown")
            # last.pt is authoritative: repair derived files after a crash between atomic replacements.
            if progress["selected"] is not None:
                save_checkpoint(folder / "best.pt", progress=progress, **args)
            write_history(folder, progress["history"])
            for epoch in range(progress["completed_epoch"], 8):
                if stop_requested():
                    break
                training = None
                in_flight_epoch = epoch + 1
                _event(folder, "epoch_started", attempt_id=attempt.name, epoch=epoch + 1,
                       committed_epoch=progress["completed_epoch"], committed_updates=progress["updates"])
                sampler = context["loaders"]["train"].sampler
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                started = time.perf_counter()
                train = train_epoch(context["model"], context["loaders"]["train"], context["optimizer"], context["scheduler"], context["scaler"],
                                    device=context["device"], runtime=recipe["runtime"], accumulation=recipe["batch"]["gradient_accumulation"], clipping=recipe["optimizer"]["gradient_clipping"])
                training = train
                _event(folder, "training_observed", attempt_id=attempt.name, epoch=epoch + 1, train=train)
                metrics = evaluate(context["model"], context["loaders"]["val"], device=context["device"], runtime=recipe["runtime"], n_train=context["n_train"], labels=context["labels"])
                if (train["samples"] != context["train_samples"] or metrics["samples"] != context["val_samples"]
                        or train["optimizer_updates"] != recipe["batch"]["updates_per_epoch"]):
                    raise ValueError("Incomplete natural epoch or full validation")
                row = {"epoch": epoch + 1, "train": train, "validation": metrics, "epoch_seconds": time.perf_counter() - started}
                candidate = {"epoch": epoch + 1, "macro_f1_supported": metrics["macro_f1_supported"], "nll": metrics["nll"]}
                if select_checkpoint(candidate, progress["selected"]):
                    progress["selected"] = candidate
                    progress["selected_weights"] = {k: v.detach().cpu().clone() for k, v in context["model"].state_dict().items()}
                progress["history"].append(row)
                progress["completed_epoch"] = epoch + 1
                progress["updates"] += train["optimizer_updates"]
                save_checkpoint(folder / "last.pt", progress=progress, **args)
                save_checkpoint(folder / "best.pt", progress=progress, **args)
                write_history(folder, progress["history"])
                _event(folder, "epoch_committed", attempt_id=attempt.name, epoch=epoch + 1, updates=progress["updates"])
                in_flight_epoch = None
                if stop_requested():
                    break
            complete = progress["completed_epoch"] == 8
            summary = {"kind": "official_e2_run_summary_v1", "status": "complete" if complete else "interrupted",
                       "run_id": cfg["run_id"], "stage": cfg["stage"], "seed": cfg["seed"], "lr": cfg["lr"],
                       "contract_sha256": CONTRACT_SHA256, "config_sha256": cfg["config_sha256"],
                       **topology(recipe), "budget": recipe["budget"],
                       "identity": checked, "evidence_kind": checked["scientific_runtime"]["evidence_kind"],
                       "planned_epochs": 8, "completed_epochs": progress["completed_epoch"], "validations": len(progress["history"]),
                       "optimizer_updates": progress["updates"], "planned_optimizer_updates": recipe["budget"]["planned_max_updates"],
                       "selected": progress["selected"], "termination_reason": "eight_epoch_budget_complete" if complete else "operational_boundary_stop",
                       "checkpoint_sha256": {p.name: sha256_file(p) for p in (folder / "best.pt", folder / "last.pt") if p.exists()},
                       "history_sha256": sha256_file(folder / "history.json"), "test_images_opened": 0}
            # Publish the complete summary last. A killed publication can resume the authoritative
            # boundary (including boundary eight) without executing another optimizer update.
            _event(folder, summary["status"], attempt_id=attempt.name, completed_epochs=progress["completed_epoch"])
            _attempt_snapshot(attempt / "outcome.json", {"status": summary["status"], "last_committed_epoch": progress["completed_epoch"],
                              "last_committed_updates": progress["updates"], "elapsed_seconds": time.perf_counter() - attempt_started})
            atomic_json(folder / "run_summary.json", summary)
            return summary
        except Exception as exc:
            boundary = _last_boundary(folder, context)
            failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                       "last_committed_epoch": boundary["completed_epoch"], "last_committed_updates": boundary["updates"],
                       "in_flight_epoch": in_flight_epoch, "observed_training": training,
                       "partial_work": "unknown" if in_flight_epoch is not None and training is None else "available measurements recorded; only committed boundaries count toward budget",
                       "elapsed_seconds": time.perf_counter() - attempt_started}
            _attempt_snapshot(attempt / ("failure.json" if (attempt / "outcome.json").exists() else "outcome.json"), failure)
            _event(folder, "failure", attempt_id=attempt.name, **failure)
            atomic_json(folder / "run_summary.json", {"status": "failed", "run_id": config["run_id"], "error_type": type(exc).__name__,
                        "error": str(exc), "resume_from_last_committed_boundary_only": True, "test_images_opened": 0})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = validate_run(read_json(args.config))
    if args.check_config:
        print("E2 declaration valid; no model/GPU allocated or job authorized")
        return
    requested = {"stop": False}
    def request_stop(*_):
        requested["stop"] = True
    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    run(config, Path(config["paths"]["runs_dir"]) / config["run_id"], resume=args.resume, stop_requested=lambda: requested["stop"])


if __name__ == "__main__":
    main()
