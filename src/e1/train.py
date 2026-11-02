"""One explicitly authorized official E1 run; no job submission or test evaluator."""
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
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import atomic_json, atomic_text, read_checkpoint, restore_checkpoint, save_checkpoint
from .config import (CONTRACT_SHA256, ROOT, fingerprint, read_json, require_approval,
                     sha256_file, validate_run)
from .data import FixedManifest, make_loaders
from .engine import UpdateScheduler, evaluate, select_checkpoint, train_epoch
from .model import build_e1, build_optimizer


def source_identity() -> dict:
    paths = sorted((ROOT / "src/e1").glob("*.py")) + sorted((ROOT / "src/preprocessing").glob("*.py"))
    paths += [ROOT / "requirements.txt", ROOT / "pyproject.toml"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Official execution requires the accepted clean Git checkout") from exc
    if dirty:
        raise ValueError("Official execution requires clean source")
    return {"git_commit": commit, "source_sha256": fingerprint(hashes)}


def _software() -> dict:
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("torch", "torchvision", "Pillow", "numpy", "pandas", "scikit-learn")},
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}


def _allocation(recipe):
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise ValueError("Official GPU work requires a Slurm allocation")
    text = subprocess.check_output(["scontrol", "show", "job", "--oneliner", job_id], text=True)
    fields = dict(word.split("=", 1) for word in text.split() if "=" in word)
    d = recipe["deployment"]
    if (fields.get("Account") != d["account"] or fields.get("QOS") != d["qos"]
            or fields.get("Partition") != d["partition"] or fields.get("TimeLimit") != d["job_walltime"]
            or os.environ.get("SLURM_CPUS_PER_TASK") != str(d["cpus_per_task"])
            or os.environ.get("SLURM_MEM_PER_NODE") != str(d["slurm_mem_mib"])
            or "gres/gpu:2g.35gb=1" not in fields.get("ReqTRES", "").split(",")):
        raise ValueError("Actual Slurm allocation differs from reviewed contract")
    return {"job_id": job_id, "record": fields}


def preflight(config: dict) -> dict:
    config = validate_run(config)
    recipe, paths = config["recipe"], config["paths"]
    needed = {"acceptance_record", "authorization_record", "conformance_report"}
    if needed - paths.keys():
        raise ValueError("G2 acceptance, stage authorization, and target conformance records required")
    acceptance = require_approval(Path(paths["acceptance_record"]), "G2")
    gate = "G3a" if config["stage"] == "tuning" else "G3b"
    approval = require_approval(Path(paths["authorization_record"]), gate, config["run_id"])
    if approval.get("acceptance_sha256") != sha256_file(Path(paths["acceptance_record"])):
        raise ValueError("Stage approval must bind the reviewed G2 acceptance record")
    if config["stage"] == "final":
        from .campaign import read_selection
        selection = read_selection(Path(paths["selection"]), recipe)
        if selection["lr"] != config["lr"] or approval.get("selection_sha256") != config["selection_sha256"]:
            raise ValueError("G3b does not authorize this locked selected configuration")
    reviewed_paths = recipe["deployment"]["paths"]
    for key, reviewed in [("data_root", "dataset"), ("weights", "pretrained"), ("runs_dir", "official_e1_runs")]:
        if paths[key] != reviewed_paths[reviewed]:
            raise ValueError("Official paths differ from reviewed target")
    expected_env = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": str(config["seed"])}
    if any(os.environ.get(k) != v for k, v in expected_env.items()) or os.environ.get("NVIDIA_TF32_OVERRIDE") not in {None, "0"}:
        raise ValueError("Reviewed environment must be exported before Python startup")
    software = _software()
    if software != recipe["runtime"]["software_lock"]:
        raise ValueError("Target software differs from reviewed lock; actual-environment acceptance required")
    allocation = _allocation(recipe)
    manifest = FixedManifest()
    pre = recipe["preprocessing"]
    if sha256_file(ROOT / pre["implementation_path"]) != pre["implementation_sha256"]:
        raise ValueError("Frozen preprocessing implementation changed")
    weight_sha = sha256_file(Path(paths["weights"]))
    if Path(paths["weights"]).name != recipe["model"]["pretrained"]["filename"] or weight_sha != recipe["model"]["pretrained"]["sha256"]:
        raise ValueError("Pinned offline ImageNet weights do not match")
    report_path = Path(paths["conformance_report"])
    report = read_json(report_path)
    if (report.get("status") != "pass" or report.get("test_images_opened") != 0
            or report.get("dataset_root") != paths["data_root"] or report.get("fingerprints") != manifest.fingerprints
            or report.get("preprocessing_sha256") != pre["config_sha256"]
            or report.get("image_hash_audit") != {"train": 34997, "val": 7497}
            or report.get("environment", {}).get("packages") != {k: software[k] for k in ("torch", "torchvision", "Pillow", "numpy", "pandas", "scikit-learn")}):
        raise ValueError("Actual root/packages need passing train/val-only target conformance")
    common = {**source_identity(), "software": software, "fingerprints": manifest.fingerprints,
              "preprocessing_sha256": pre["config_sha256"], "pretrained_sha256": weight_sha,
              "dataset_root": paths["data_root"], "conformance_sha256": sha256_file(report_path),
              "gpu_class": recipe["deployment"]["expected_gpu_class"]}
    if acceptance.get("identity") != common:
        raise ValueError("Actual source/data/environment differs from reviewed G2 acceptance")
    # A historical audit does not prove that files or symlink targets stayed immutable.
    # Recheck train/val bytes and containment once before allocation, never every epoch.
    manifest.verify_images(Path(paths["data_root"]))
    return {**common, "contract_sha256": CONTRACT_SHA256, "evidence_kind": "official_uit_slurm",
            "accepted_device": acceptance["device"], "acceptance_sha256": sha256_file(Path(paths["acceptance_record"]))}


def _setup(config, checked):
    recipe = config["recipe"]
    if not torch.cuda.is_available():
        raise ValueError("Official E1 runs require the accepted UIT CUDA environment")
    random.seed(config["seed"])
    np.random.seed(config["seed"] % 2**32)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    device = {"name": props.name, "total_bytes": props.total_memory, "driver": driver}
    if props.name != recipe["deployment"]["expected_gpu_class"] or device != checked["accepted_device"]:
        raise ValueError("Actual device/driver requires renewed target acceptance; UUID is not pinned")
    model, metadata = build_e1(Path(config["paths"]["weights"]), checked["pretrained_sha256"], {**recipe["initialization"], "seed": config["seed"]})
    model.to("cuda")
    optimizer, groups = build_optimizer(model, {**recipe["optimizer"], "lr": config["lr"]})
    manifest = FixedManifest()
    loaders = make_loaders(manifest, Path(config["paths"]["data_root"]), config)
    if math.ceil(len(loaders["train"]) / recipe["batch"]["gradient_accumulation"]) != recipe["batch"]["updates_per_epoch"]:
        raise ValueError("Natural loader and reviewed update accounting differ")
    scheduler = UpdateScheduler(optimizer, recipe["scheduler"], recipe["budget"]["planned_max_updates"])
    actual = {"cpu_threads": torch.get_num_threads(), "matmul_precision": torch.get_float32_matmul_precision(),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}
    expected = {"cpu_threads": 2, "matmul_precision": "highest", "tf32_matmul": False, "tf32_cudnn": False,
                "cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True, "warn_only": False}
    if actual != expected:
        raise ValueError("Effective Torch runtime differs from reviewed policy")
    return dict(config=config, model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, loaders=loaders,
                device=torch.device("cuda"), n_train=manifest.n_train_by_label, labels=manifest.labels,
                train_samples=manifest.n_train, val_samples=len(manifest.records("val")), identity=checked,
                environment={"software": checked["software"], "device": device, "allocation": _allocation(recipe),
                             "actual_torch_settings": actual, "runtime_policy": recipe["runtime"],
                             "runtime_environment": {k: os.environ.get(k) for k in recipe["runtime"]["environment"]},
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
                model_metadata={**metadata, "optimizer_groups": groups})


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
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _begin_attempt(folder, config, operation):
    root = folder / "attempts"
    root.mkdir(exist_ok=True)
    previous = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    for old in previous:
        if not (old / "outcome.json").exists() and not (old / "abandoned.json").exists():
            observation = {"status": "terminated_without_outcome", "uncommitted_work": "unknown after abrupt termination",
                           "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
            _attempt_snapshot(old / "abandoned.json", observation)
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
            checked = preflight(config)
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
            summary = {"kind": "official_e1_run_summary_v1", "status": "complete" if complete else "interrupted",
                       "run_id": cfg["run_id"], "stage": cfg["stage"], "seed": cfg["seed"], "lr": cfg["lr"],
                       "contract_sha256": CONTRACT_SHA256, "config_sha256": cfg["config_sha256"],
                       "identity": checked, "evidence_kind": checked["evidence_kind"],
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
        print("E1 declaration valid; no model/GPU allocated or job authorized")
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
