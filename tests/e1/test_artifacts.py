"""Fabricated trusted record envelopes exercise parsing, never actual UIT evidence.

Every artifact lives in pytest's temporary directory. No target workload is run;
the simulated approvals/checkpoints are not exported as official experiments.
"""
import copy
import csv
import json

import pytest
import torch


@pytest.fixture
def completed_trials(recipe, tmp_path):
    from src.e1.checkpoint import atomic_json, save_checkpoint
    from src.e1.config import CONTRACT_SHA256, ROOT, resolve_run, sha256_file
    from src.e1.engine import UpdateScheduler
    from src.e1.metrics import Metrics124
    from torch.utils.data import DataLoader
    from .test_engine import Tiny, tiny_data
    paths = {"data_root": str(tmp_path / "simulated-data"), "weights": str(tmp_path / "convnext_tiny-983f1562.pth"),
             "runs_dir": str(tmp_path / "simulated-runs"), "acceptance_record": str(tmp_path / "simulated-g2.json"),
             "authorization_record": str(tmp_path / "simulated-g3a.json"), "conformance_report": str(tmp_path / "simulated-conformance.json")}
    atomic_json(tmp_path / "fixture_manifest.json", {"kind": "synthetic_artifact_parser_fixture", "real_gpu_work": False})
    atomic_json(tmp_path / "simulated-conformance.json", {"simulation_only": True})
    common = {"git_commit": "a"*40, "source_sha256": "a"*64, "software": recipe["runtime"]["software_lock"],
              "fingerprints": {k: v["sha256"] for k, v in recipe["data"]["artifacts"].items()},
              "preprocessing_sha256": recipe["preprocessing"]["config_sha256"], "pretrained_sha256": recipe["model"]["pretrained"]["sha256"],
              "dataset_root": paths["data_root"], "conformance_sha256": sha256_file(tmp_path / "simulated-conformance.json"),
              "gpu_class": recipe["deployment"]["expected_gpu_class"]}
    device = {"name": recipe["deployment"]["expected_gpu_class"], "driver": "580.126.20", "total_bytes": 34896609280}
    atomic_json(tmp_path / "simulated-g2.json", {"gate": "G2", "approved": True, "contract_sha256": CONTRACT_SHA256,
                "identity": common, "device": device})
    configs = [resolve_run(recipe, stage="tuning", lr=lr, seed=1001, paths=paths) for lr in recipe["campaign"]["lr_candidates"]]
    acceptance_sha = sha256_file(tmp_path / "simulated-g2.json")
    atomic_json(tmp_path / "simulated-g3a.json", {"gate": "G3a", "approved": True, "contract_sha256": CONTRACT_SHA256,
                "acceptance_sha256": acceptance_sha, "run_ids": [c["run_id"] for c in configs]})
    identity = {**common, "contract_sha256": CONTRACT_SHA256, "evidence_kind": "official_uit_slurm",
                "accepted_device": device, "acceptance_sha256": acceptance_sha}
    # Build metric counts from the fixed metadata; no images are opened.
    with (ROOT / "artifacts/baseline124/class_support.csv").open(encoding="utf-8", newline="") as stream:
        classes = list(csv.DictReader(stream))
    targets = torch.tensor([int(r["label_id"]) for r in classes for _ in range(int(r["val"]))])
    logits = torch.zeros(len(targets), 124)
    logits[torch.arange(len(targets)), targets] = 100
    metric = Metrics124()
    metric.update(logits, targets, 0.)
    metrics = metric.compute({int(r["label_id"]): int(r["train"]) for r in classes}, [r["label"] for r in classes])
    folders = []
    for config in configs:
        folder = tmp_path / "simulated-runs" / config["run_id"]
        folder.mkdir(parents=True)
        atomic_json(folder / "config_resolved.json", config)
        model = Tiny()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        scheduler = UpdateScheduler(optimizer, recipe["scheduler"], 139992)
        state = scheduler.state_dict()
        state["completed"] = 139992
        scheduler.load_state_dict(state)
        history = [{"epoch": e, "train": {"samples": 34997, "optimizer_updates": 17499}, "validation": copy.deepcopy(metrics)} for e in range(1, 9)]
        selected = {"epoch": 1, "macro_f1_supported": 1., "nll": 0.}
        progress = {"completed_epoch": 8, "updates": 139992, "history": history, "selected": selected,
                    "selected_weights": copy.deepcopy(model.state_dict())}
        args = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, config=config, progress=progress,
                    identity=identity, loaders={"train": DataLoader(tiny_data(), generator=torch.Generator().manual_seed(99))})
        save_checkpoint(folder / "last.pt", **args)
        save_checkpoint(folder / "best.pt", **args)
        atomic_json(folder / "history.json", history)
        atomic_json(folder / "events.json", [{"event": "created"}, {"event": "complete"}])
        summary = {"kind": "official_e1_run_summary_v1", "status": "complete", "evidence_kind": "official_uit_slurm",
                   "contract_sha256": CONTRACT_SHA256, "test_images_opened": 0, "identity": identity,
                   **{k: config[k] for k in ("run_id", "stage", "seed", "lr", "config_sha256")},
                   "planned_epochs": 8, "completed_epochs": 8, "validations": 8, "optimizer_updates": 139992,
                   "planned_optimizer_updates": 139992, "termination_reason": "eight_epoch_budget_complete", "selected": selected,
                   "checkpoint_sha256": {n: sha256_file(folder / n) for n in ("best.pt", "last.pt")}, "history_sha256": sha256_file(folder / "history.json")}
        atomic_json(folder / "run_summary.json", summary)
        folders.append(folder)
    return folders


def test_real_completed_evidence_parser_and_locked_selection_success_path(recipe, completed_trials, tmp_path):
    from src.e1.campaign import validate_completed_run, select_lr, write_selection, read_selection
    for folder in completed_trials:
        row = validate_completed_run(folder)
        assert row["summary"]["completed_epochs"] == 8
    selection = select_lr(recipe, [{"run_dir": str(p)} for p in completed_trials])
    assert selection["lr"] == 2.5e-5
    p = tmp_path / "selection.json"
    write_selection(p, recipe, selection)
    assert read_selection(p, recipe) == selection


def test_rehashed_incomplete_history_is_still_rejected(completed_trials):
    from src.e1.campaign import validate_completed_run
    from src.e1.checkpoint import atomic_json
    from src.e1.config import read_json, sha256_file
    folder = completed_trials[0]
    atomic_json(folder / "history.json", read_json(folder / "history.json")[:7])
    summary = read_json(folder / "run_summary.json")
    summary["history_sha256"] = sha256_file(folder / "history.json")
    atomic_json(folder / "run_summary.json", summary)
    with pytest.raises(ValueError, match="history"):
        validate_completed_run(folder)


def test_rehashed_wrong_selected_weights_are_still_rejected(completed_trials):
    from src.e1.campaign import validate_completed_run
    from src.e1.checkpoint import atomic_json
    from src.e1.config import read_json, sha256_file
    folder = completed_trials[0]
    state = torch.load(folder / "best.pt", weights_only=False)
    first = next(iter(state["model"]))
    state["model"][first] = state["model"][first] + 1
    torch.save(state, folder / "best.pt")
    summary = read_json(folder / "run_summary.json")
    summary["checkpoint_sha256"]["best.pt"] = sha256_file(folder / "best.pt")
    atomic_json(folder / "run_summary.json", summary)
    with pytest.raises(ValueError, match="lineage"):
        validate_completed_run(folder)
