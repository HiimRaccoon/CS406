"""End-to-end control flow with tiny synthetic CPU data; no real pilot run."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import json
import sys

import pytest
import torch
from torch.utils.data import Dataset

from src.pilot import p0_feasibility as p0, p1_train as p1
from src.pilot.runtime import create_run
from test_checkpoint_engine import TinyPilot
from test_config import complete_config


class SyntheticDataset(Dataset):
    def __init__(self, manifest, root, split, processor, seed, augmentation=None):
        self.split = split

    def __len__(self):
        return 7 if self.split == "train" else 5

    def __getitem__(self, index):
        _, i = index if isinstance(index, tuple) else (0, index)
        return {"global": torch.tensor([.1, .2, .3, .4]) + i * .01}, i % 3


def setup_runner(monkeypatch, module, manifest):
    cpu_torch = SimpleNamespace(device=lambda _: torch.device("cpu"),
                                cuda=SimpleNamespace(reset_peak_memory_stats=lambda _: None,
                                                     max_memory_allocated=lambda _: 0,
                                                     max_memory_reserved=lambda _: 0,
                                                     OutOfMemoryError=torch.cuda.OutOfMemoryError))
    monkeypatch.setattr(module, "torch", cpu_torch)
    monkeypatch.setattr(module, "FixedManifest", lambda _: manifest)
    monkeypatch.setattr(module, "apply_runtime", lambda _: torch.manual_seed(37))
    monkeypatch.setattr(module, "environment", lambda _: {"synthetic_cpu_test": True})
    monkeypatch.setattr(module, "identity", lambda *args: {"synthetic_cpu_test": True})
    monkeypatch.setattr(module, "validate_gates", lambda *args: [{"identity": {"synthetic_cpu_test": True}}] * 3)
    monkeypatch.setattr(module, "build_model", lambda *args: (TinyPilot(.2), {"synthetic": True}))
    monkeypatch.setattr(module, "PilotDataset", SyntheticDataset)


def test_p0_full_epoch_and_original_image_count(monkeypatch, tmp_path, manifest):
    setup_runner(monkeypatch, p0, manifest)
    cfg = complete_config()
    cfg["paths"]["runs_dir"] = str(tmp_path)
    weights = tmp_path / "convnext_tiny-983f1562.pth"
    weights.write_bytes(b"synthetic CPU integration fixture; never pretrained")
    cfg["paths"]["weights"] = str(weights)
    cfg["expected_weight_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    # Four microbatches available: one warmup, one measured, separate full epoch.
    folder = create_run(cfg, "synthetic-p0")
    result = p0.run(cfg, folder)
    assert result["status"] == "pass"
    assert result["timing"]["steady_state"]["samples"] == 2
    assert result["timing"]["full_epoch"]["samples"] == 7
    assert result["test_images_opened"] == 0


@pytest.mark.parametrize("kind", ["oom", "criteria"])
def test_p0_failure_summary_preserves_verified_fallback_context(monkeypatch, tmp_path, manifest, kind):
    setup_runner(monkeypatch, p0, manifest)
    cfg = complete_config(exp="e1hr")
    cfg["paths"]["runs_dir"] = str(tmp_path)
    weights = tmp_path / "convnext_tiny-983f1562.pth"
    weights.write_bytes(b"synthetic CPU failure test; never pretrained")
    cfg["paths"]["weights"] = str(weights)
    cfg["expected_weight_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    cfg["resolution_policy"]["ordered_candidates"].append([1024, 448])
    cfg["resolution_policy"]["batch_policy"]["candidate_batches"].append({"microbatch": 1, "accumulation": 2})
    if kind == "oom":
        def injected_allocation_failure(*args):
            raise torch.cuda.OutOfMemoryError("synthetic exception; no GPU allocation")
        monkeypatch.setattr(p0, "build_model", injected_allocation_failure)
        error = torch.cuda.OutOfMemoryError
    else:
        cfg["feasibility"]["min_images_per_second"] = 1e30
        error = SystemExit
    monkeypatch.setattr(p0, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(sys, "argv", ["pilot", "--config", "unused", "--run-id", "failure-evidence"])
    with pytest.raises(error):
        p0.main()
    summary_path = tmp_path / "failure-evidence/run_summary.json"
    report = json.loads(summary_path.read_text())
    assert report["failure_kind"] == kind
    context = report["feasibility_context"]
    assert context["protocol_id"] == "unit-test-only"
    assert context["pretrained_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert context["fingerprints"] == manifest.fingerprints
    assert context["runtime"] == cfg["runtime"] and context["initialization"] == cfg["initialization"]
    fallback = copy.deepcopy(cfg)
    fallback["global_wh"] = [1024, 448]
    fallback["batch"] = {"microbatch": 1, "accumulation": 2}
    fallback["resolution_policy"]["prior_failure_reports"] = [str(summary_path)]
    p0.validate_resolution_evidence(fallback, context)
    report["batch"]["microbatch"] = 64
    summary_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="batch"):
        p0.validate_resolution_evidence(fallback, context)


def test_p1_bounded_history_and_resumption(monkeypatch, tmp_path, manifest):
    setup_runner(monkeypatch, p1, manifest)
    cfg = complete_config("p1")
    cfg["paths"]["runs_dir"] = str(tmp_path)
    full = create_run(cfg, "synthetic-full")
    expected = p1.run(cfg, full)
    assert expected["completed_epochs"] == expected["planned_epochs"] == 2
    assert expected["optimizer_updates"] == expected["planned_optimizer_updates"] == 8
    interrupted = create_run(cfg, "synthetic-resume")
    # Stop callback: before epoch 1 false, after epoch 1 true.
    calls = iter([False, True])
    prefix = p1.run(cfg, interrupted, stop_requested=lambda: next(calls))
    assert prefix["status"] == "interrupted_at_epoch_boundary"
    checkpoint = interrupted / "checkpoint_last.pt"
    create_run(cfg, resume=checkpoint)
    resumed = p1.run(cfg, interrupted, resume=checkpoint)
    assert resumed["status"] == "complete" and resumed["completed_epochs"] == 2
    a = torch.load(full / "checkpoint_last.pt", weights_only=False)
    b = torch.load(checkpoint, weights_only=False)
    for key, value in a["model"].items():
        assert torch.equal(value, b["model"][key])
    assert a["scheduler"] == b["scheduler"]
    for name in ("metrics.csv", "learning_curves.svg", "history.json", "timing.json", "memory.json"):
        assert (interrupted / name).is_file()
