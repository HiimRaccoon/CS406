"""End-to-end control flow with tiny synthetic CPU data; no real pilot run."""
from pathlib import Path
from types import SimpleNamespace

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
                                                     max_memory_reserved=lambda _: 0))
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
    # Four microbatches available: one warmup, one measured, separate full epoch.
    folder = create_run(cfg, "synthetic-p0")
    result = p0.run(cfg, folder)
    assert result["status"] == "pass"
    assert result["timing"]["steady_state"]["samples"] == 2
    assert result["timing"]["full_epoch"]["samples"] == 7
    assert result["test_images_opened"] == 0


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
