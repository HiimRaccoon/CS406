import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader

from .test_engine import Tiny, tiny_data


def install_cpu_edges(monkeypatch):
    import src.e1.train as module
    from src.e1.engine import UpdateScheduler
    identity = {"evidence_kind": "synthetic_cpu_fixture"}
    monkeypatch.setattr(module, "preflight", lambda config: identity)
    def setup(config, checked):
        torch.manual_seed(config["seed"])
        c = copy.deepcopy(config)
        # Private integration fixture only; production _setup never substitutes reviewed counts.
        c["recipe"]["batch"]["updates_per_epoch"] = 4
        c["recipe"]["budget"]["planned_max_updates"] = 32
        m = Tiny(.2)
        optimizer = torch.optim.AdamW(m.parameters(), lr=c["lr"])
        scheduler = UpdateScheduler(optimizer, {"warmup_updates": 2, "warmup_start_factor": .1, "cosine_floor_lr": 1e-6}, 32)
        loaders = {s: DataLoader(tiny_data(n), batch_size=2, generator=torch.Generator().manual_seed(71)) for s, n in [("train", 7), ("val", 5)]}
        return dict(config=c, model=m, optimizer=optimizer, scheduler=scheduler, scaler=None, loaders=loaders,
                    device=torch.device("cpu"), n_train={i: 1 for i in range(124)}, labels=[str(i) for i in range(124)],
                    train_samples=7, val_samples=5, identity=checked, environment={"synthetic": True}, model_metadata={})
    monkeypatch.setattr(module, "_setup", setup)
    return module


@pytest.mark.parametrize("slope", [0., .01])
def test_flat_and_decreasing_scores_still_finish_all_eight_epochs(run_config, monkeypatch, slope):
    module = install_cpu_edges(monkeypatch)
    real = module.evaluate
    calls = []
    def decreasing(*a, **kw):
        metrics = real(*a, **kw)
        calls.append(len(calls)+1)
        metrics["macro_f1_supported"] = .5 - slope*len(calls)
        return metrics
    monkeypatch.setattr(module, "evaluate", decreasing)
    folder = __import__('pathlib').Path(run_config["paths"]["runs_dir"]) / run_config["run_id"]
    result = module.run(run_config, folder)
    assert result["status"] == "complete" and result["completed_epochs"] == result["validations"] == 8
    assert result["optimizer_updates"] == result["planned_optimizer_updates"] == 32
    assert result["selected"]["epoch"] == 1 and calls == list(range(1, 9))
    best = torch.load(folder / "best.pt", weights_only=False)
    last = torch.load(folder / "last.pt", weights_only=False)
    assert best["progress"]["selected"]["epoch"] == 1
    assert any(not torch.equal(best["model"][k], last["model"][k]) for k in best["model"])
    assert len(json.loads((folder / "history.json").read_text())) == 8
    assert result["evidence_kind"] == "synthetic_cpu_fixture"
    with pytest.raises((FileExistsError, ValueError)):
        module.run(run_config, folder)


def test_epoch_zero_and_partial_epoch_replay_match_uninterrupted(run_config, monkeypatch, tmp_path):
    module = install_cpu_edges(monkeypatch)
    from src.e1.config import resolve_run
    recipe = run_config["recipe"]
    configs = [resolve_run(recipe, stage="tuning", lr=1e-4, seed=1001,
                           paths={**run_config["paths"], "runs_dir": str(tmp_path / name)}) for name in ("full", "replay")]
    full, interrupted = [__import__('pathlib').Path(c["paths"]["runs_dir"]) / c["run_id"] for c in configs]
    module.run(configs[0], full)
    original = module.train_epoch
    def broken(*a, **kw):
        original(*a, **kw)
        raise RuntimeError("injected partial-epoch failure after work")
    monkeypatch.setattr(module, "train_epoch", broken)
    with pytest.raises(RuntimeError):
        module.run(configs[1], interrupted)
    assert torch.load(interrupted / "last.pt", weights_only=False)["progress"]["completed_epoch"] == 0
    monkeypatch.setattr(module, "train_epoch", original)
    calls = iter([False, True])
    prefix = module.run(configs[1], interrupted, resume=interrupted / "last.pt", stop_requested=lambda: next(calls))
    assert prefix["status"] == "interrupted" and prefix["completed_epochs"] == 1
    module.run(configs[1], interrupted, resume=interrupted / "last.pt")
    a, b = [torch.load(p / "last.pt", weights_only=False) for p in (full, interrupted)]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k])
    assert a["scheduler"] == b["scheduler"]
    events = json.loads((interrupted / "events.json").read_text())
    assert any(e["event"] == "failure" for e in events) and sum(e["event"] == "resume" for e in events) == 2


def test_preflight_without_execution_approvals_fails_before_model_allocation(run_config, monkeypatch):
    import src.e1.train as module
    monkeypatch.setattr(module, "build_e1", lambda *a: pytest.fail("model allocated"))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.preflight(run_config)


def test_run_lock_rejects_second_writer(tmp_path):
    from src.e1.train import run_lock
    with run_lock(tmp_path):
        with pytest.raises((OSError, BlockingIOError)):
            with run_lock(tmp_path):
                pytest.fail("second writer acquired lock")
