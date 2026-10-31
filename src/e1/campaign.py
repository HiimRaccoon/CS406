"""Four-trial evidence ledger and locked LR selection; never submits jobs."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from .checkpoint import read_checkpoint
from .config import (CONTRACT_SHA256, fingerprint, load_recipe, read_json, require_approval,
                     resolve_run, sha256_file, validate_recipe, validate_run)
from .engine import select_checkpoint


def plan_tuning(recipe: dict, paths: dict) -> list[dict]:
    recipe = validate_recipe(recipe)
    return [resolve_run(recipe, stage="tuning", lr=lr, seed=recipe["campaign"]["tuning_seed"], paths=paths)
            for lr in recipe["campaign"]["lr_candidates"]]


def _write_new(path: Path, value):
    # Exclusive creation preserves reviewed declarations/selections instead of silently replacing them.
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_declarations(output: Path, declarations: list[dict]) -> None:
    if not declarations or len({d["run_id"] for d in declarations}) != len(declarations):
        raise ValueError("Unique complete declarations required")
    declarations = [validate_run(d) for d in declarations]
    stage = declarations[0]["stage"]
    if any(d["stage"] != stage for d in declarations) or len(declarations) != (4 if stage == "tuning" else 3):
        raise ValueError("Wrong declaration stage/count")
    if stage == "tuning" and {d["lr"] for d in declarations} != {2.5e-5, 5e-5, 1e-4, 2e-4}:
        raise ValueError("Exactly four reviewed tuning LRs required")
    if stage == "final" and ({d["seed"] for d in declarations} != {2001, 3001, 4001} or len({d["lr"] for d in declarations}) != 1):
        raise ValueError("Exactly the three fresh final seeds at one selected LR required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for declaration in declarations:
        _write_new(output / (declaration["run_id"] + ".json"), declaration)
    _write_new(output / "ledger.json", {"kind": "e1_campaign_declarations_v1", "status": "declared_not_executed",
               "contract_sha256": CONTRACT_SHA256, "stage": stage,
               "runs": [{"run_id": d["run_id"], "config_sha256": d["config_sha256"]} for d in declarations],
               "execution_authorized": False})


def validate_completed_run(run_dir: Path) -> dict:
    folder = Path(run_dir).resolve()
    try:
        summary = read_json(folder / "run_summary.json")
        if (summary.get("kind") != "official_e1_run_summary_v1" or summary.get("status") != "complete"
                or summary.get("evidence_kind") != "official_uit_slurm" or summary.get("test_images_opened") != 0
                or summary.get("contract_sha256") != CONTRACT_SHA256):
            raise ValueError("Only valid completed official UIT E1 evidence is eligible")
        config = validate_run(read_json(folder / "config_resolved.json"))
        recipe, identity = config["recipe"], summary["identity"]
        if (folder.name != config["run_id"] or folder != (Path(config["paths"]["runs_dir"]) / config["run_id"]).resolve()
                or any(summary[k] != config[k] for k in ("run_id", "stage", "seed", "lr", "config_sha256"))
                or summary["planned_epochs"] != 8 or summary["completed_epochs"] != 8 or summary["validations"] != 8
                or summary["optimizer_updates"] != 139992 or summary["planned_optimizer_updates"] != 139992
                or summary["termination_reason"] != "eight_epoch_budget_complete"):
            raise ValueError("Wrong run identity or incomplete eight-epoch budget")
        acceptance_path = Path(config["paths"]["acceptance_record"])
        acceptance = require_approval(acceptance_path, "G2")
        expected_identity = {**acceptance["identity"], "contract_sha256": CONTRACT_SHA256, "evidence_kind": "official_uit_slurm",
                             "accepted_device": acceptance["device"], "acceptance_sha256": sha256_file(acceptance_path)}
        if identity != expected_identity or identity["software"] != recipe["runtime"]["software_lock"]:
            raise ValueError("Run does not match accepted UIT environment/source identity")
        if (identity["fingerprints"] != {k: v["sha256"] for k, v in recipe["data"]["artifacts"].items()}
                or identity["pretrained_sha256"] != recipe["model"]["pretrained"]["sha256"]
                or identity["preprocessing_sha256"] != recipe["preprocessing"]["config_sha256"]
                or identity["gpu_class"] != recipe["deployment"]["expected_gpu_class"]):
            raise ValueError("Wrong accepted data/weight/preprocessing/GPU identity")
        gate = "G3a" if config["stage"] == "tuning" else "G3b"
        approval = require_approval(Path(config["paths"]["authorization_record"]), gate, config["run_id"])
        if approval.get("acceptance_sha256") != identity["acceptance_sha256"]:
            raise ValueError("Stage authorization does not bind accepted environment")
        if config["stage"] == "final" and approval.get("selection_sha256") != config["selection_sha256"]:
            raise ValueError("Final run lacks G3b binding to selected configuration")
        for name in ("best.pt", "last.pt"):
            if sha256_file(folder / name) != summary["checkpoint_sha256"][name]:
                raise ValueError("Checkpoint hash mismatch")
        if sha256_file(folder / "history.json") != summary["history_sha256"]:
            raise ValueError("History hash mismatch")
        history = read_json(folder / "history.json")
        if len(history) != 8:
            raise ValueError("Missing full validation history")
        selected = None
        for epoch, row in enumerate(history, 1):
            train, val = row["train"], row["validation"]
            if (row["epoch"] != epoch or train["samples"] != 34997 or train["optimizer_updates"] != 17499
                    or val["samples"] != 7497 or val["supported_classes"] != 106 or val["output_classes"] != 124
                    or any(not math.isfinite(val[k]) or not 0 <= val[k] <= 1 for k in ("top1", "top3", "top5", "macro_f1_supported", "weighted_f1"))):
                raise ValueError("Incomplete or invalid epoch evidence")
            candidate = {"epoch": epoch, "macro_f1_supported": val["macro_f1_supported"], "nll": val["nll"]}
            if select_checkpoint(candidate, selected):
                selected = candidate
        if selected != summary["selected"]:
            raise ValueError("Selected metric/epoch does not follow frozen chronological comparator")
        last = read_checkpoint(folder / "last.pt", config=config, identity=identity)
        best = read_checkpoint(folder / "best.pt", config=config, identity=identity)
        if (last["role"] != "last" or best["role"] != "best" or last["progress"]["completed_epoch"] != 8
                or last["progress"]["history"] != history or best["progress"]["selected"] != selected
                or last["progress"]["selected"] != selected or set(best["model"]) != set(last["progress"]["selected_weights"])
                or any(not torch.equal(v, last["progress"]["selected_weights"][k]) for k, v in best["model"].items())):
            raise ValueError("Selected checkpoint lineage or committed history mismatch")
        events = read_json(folder / "events.json")
        if not isinstance(events, list) or not events or events[-1]["event"] != "complete":
            raise ValueError("Missing terminal operational evidence")
        return {"run_dir": str(folder), "config": config, "summary": summary,
                "summary_sha256": sha256_file(folder / "run_summary.json"), "best_sha256": summary["checkpoint_sha256"]["best.pt"],
                "events_sha256": sha256_file(folder / "events.json")}
    except (KeyError, TypeError, FileNotFoundError) as exc:
        raise ValueError("Incomplete official run evidence") from exc


def select_lr(recipe: dict, trial_records: list[dict]) -> dict:
    recipe = validate_recipe(recipe)
    if len(trial_records) != 4 or any(set(r) != {"run_dir"} for r in trial_records):
        raise ValueError("Require exactly four explicit trial directories, never caller-supplied scores")
    rows = [validate_completed_run(Path(r["run_dir"])) for r in trial_records]
    rows.sort(key=lambda r: r["summary"]["lr"])
    if ({r["summary"]["lr"] for r in rows} != set(recipe["campaign"]["lr_candidates"])
            or len({r["run_dir"] for r in rows}) != 4
            or any(r["summary"]["seed"] != 1001 or r["summary"]["stage"] != "tuning" for r in rows)):
        raise ValueError("Missing/repeated/wrong-identity candidate; stop for user review")
    maximum = max(r["summary"]["selected"]["macro_f1_supported"] for r in rows)
    if not math.isfinite(maximum):
        raise ValueError("Nonfinite trial metric")
    lr = min(r["summary"]["lr"] for r in rows if maximum - r["summary"]["selected"]["macro_f1_supported"] <= 1e-6)
    selection = {"kind": "e1_lr_selection_v1", "contract_sha256": CONTRACT_SHA256, "lr": lr, "tuning_seed": 1001,
                 "maximum_supported_f1": maximum, "tolerance": 1e-6, "tie_break": "lowest_base_lr_within_global_max_tolerance",
                 "trials": [{k: r[k] for k in ("run_dir", "summary_sha256", "best_sha256", "events_sha256")} for r in rows],
                 "next_stage": "STOP_FOR_USER_G3b_REVIEW", "final_execution_authorized": False}
    selection["selection_fingerprint"] = fingerprint(selection)
    return selection


def _verify_selection(recipe, selection):
    try:
        expected = select_lr(recipe, [{"run_dir": r["run_dir"]} for r in selection["trials"]])
        if fingerprint(expected) != fingerprint(selection):
            raise ValueError("Modified or stale LR selection")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete locked LR selection") from exc


def write_selection(path: Path, recipe: dict, selection: dict) -> None:
    _verify_selection(recipe, selection)
    _write_new(Path(path), selection)


def read_selection(path: Path, recipe: dict) -> dict:
    selection = read_json(path)
    _verify_selection(recipe, selection)
    return selection


def plan_final(recipe: dict, selection: dict, paths: dict) -> list[dict]:
    recipe = validate_recipe(recipe)
    if "selection" not in paths or read_selection(Path(paths["selection"]), recipe) != selection:
        raise ValueError("Final declarations require the exact persisted locked selection")
    return [resolve_run(recipe, stage="final", lr=selection["lr"], seed=seed, paths=paths) for seed in recipe["campaign"]["final_seeds"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare-tuning", "select-lr", "declare-final"))
    parser.add_argument("--input", required=True, type=Path, help="Paths JSON for declarations; list of run_dir records for selection")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    recipe, inputs = load_recipe(), read_json(args.input)
    if args.action == "declare-tuning":
        write_declarations(args.output, plan_tuning(recipe, inputs))
    elif args.action == "select-lr":
        write_selection(args.output, recipe, select_lr(recipe, inputs))
        print("STOP: hand off tuning ledger and locked selection for explicit G3b review")
    else:
        if args.selection is None:
            parser.error("--selection is required")
        inputs["selection"] = str(args.selection)
        write_declarations(args.output, plan_final(recipe, read_selection(args.selection, recipe), inputs))
    print("No jobs submitted or training authorized by this command")


if __name__ == "__main__":
    main()
