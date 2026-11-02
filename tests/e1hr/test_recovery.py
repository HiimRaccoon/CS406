"""Regressions from the independent T01-T09 review; all execution is synthetic CPU."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .test_checkpoint import components
from .test_engine import Tiny, tiny_data
from .test_train import install_cpu_edges


@pytest.mark.parametrize("failure", ["preflight", "boundary_zero"])
def test_unstarted_run_recovers_same_declaration_without_erasing_failure(run_config, monkeypatch, failure):
    module = install_cpu_edges(monkeypatch)
    name = "preflight" if failure == "preflight" else "save_checkpoint"
    original = getattr(module, name)
    def broken(*a, **kw):
        raise RuntimeError("injected failure before committed boundary zero")
    monkeypatch.setattr(module, name, broken)
    folder = Path(run_config["paths"]["runs_dir"]) / run_config["run_id"]
    with pytest.raises(RuntimeError):
        module.run(run_config, folder)
    assert not (folder / "last.pt").exists()
    monkeypatch.setattr(module, name, original)
    # Same declaration may initialize again only because no training boundary ever existed.
    result = module.run(run_config, folder)
    assert result["completed_epochs"] == 8
    events = json.loads((folder / "events.json").read_text())
    assert any(e["event"] == "failure" for e in events)


def test_terminal_publication_crash_recovers_without_training_more(run_config, monkeypatch):
    module = install_cpu_edges(monkeypatch)
    event = module._event
    class HardKill(BaseException):
        pass
    def interrupted(folder, name, **kw):
        if name == "complete":
            raise HardKill("kill at terminal publication")
        return event(folder, name, **kw)
    monkeypatch.setattr(module, "_event", interrupted)
    folder = Path(run_config["paths"]["runs_dir"]) / run_config["run_id"]
    with pytest.raises(HardKill):
        module.run(run_config, folder)
    assert torch.load(folder / "last.pt", weights_only=False)["progress"]["completed_epoch"] == 8
    monkeypatch.setattr(module, "_event", event)
    monkeypatch.setattr(module, "train_epoch", lambda *a, **k: pytest.fail("trained after eight committed epochs"))
    result = module.run(run_config, folder, resume=folder / "last.pt")
    assert result["status"] == "complete" and result["optimizer_updates"] == 32
    assert json.loads((folder / "events.json").read_text())[-1]["event"] == "complete"


def test_persistent_worker_resume_preserves_loader_model_optimizer_and_rng(run_config, tmp_path):
    import src.e1hr.data as data
    from src.e1hr.checkpoint import restore_checkpoint, save_checkpoint
    from src.e1hr.engine import train_epoch
    Loader = getattr(data, "HRLoader", DataLoader)
    torch.manual_seed(16)
    config, model, opt, sched, _ = components(run_config)
    def loader():
        return Loader(tiny_data(), batch_size=2, generator=torch.Generator().manual_seed(99), num_workers=2, persistent_workers=True)
    a = loader()
    kwargs = dict(device=torch.device("cpu"), runtime={"dtype": "float32"}, accumulation=1, clipping={"enabled": False})
    row = train_epoch(model, a, opt, sched, None, **kwargs)
    progress = {"completed_epoch": 1, "updates": 4, "history": [{"epoch": 1, "train": row}],
                "selected": {"epoch": 1, "macro_f1_supported": .5}, "selected_weights": copy.deepcopy(model.state_dict())}
    identity = {"scientific_runtime": {"evidence_kind": "synthetic_cpu_fixture"}, "audit": {"clean_git_commit": "a"*40}}
    p = tmp_path / "last.pt"
    save_checkpoint(p, model=model, optimizer=opt, scheduler=sched, scaler=None, config=config, progress=progress, identity=identity, loaders={"train": a})
    train_epoch(model, a, opt, sched, None, **kwargs)
    expected = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), sched.state_dict(), torch.get_rng_state(), a.generator.get_state()
    _, other, oo, ss, _ = components(run_config)
    b = loader()
    restore_checkpoint(p, model=other, optimizer=oo, scheduler=ss, scaler=None, config=config, identity=identity, loaders={"train": b})
    train_epoch(other, b, oo, ss, None, **kwargs)
    assert torch.equal(expected[4], b.generator.get_state())
    assert torch.equal(expected[3], torch.get_rng_state()) and expected[2] == ss.state_dict()
    for k, v in expected[0].items():
        assert torch.equal(v, other.state_dict()[k])
    for k, state in expected[1]["state"].items():
        for field, v in state.items():
            assert torch.equal(v, oo.state_dict()["state"][k][field])



def test_attempt_allocations_and_replayed_training_are_retained(run_config, monkeypatch):
    module = install_cpu_edges(monkeypatch)
    setup, evaluate = module._setup, module.evaluate
    attempts = []
    def allocated(c, checked):
        context = setup(c, checked)
        attempts.append(len(attempts) + 1)
        context["environment"] = {"allocation": {"job_id": attempts[-1]}}
        return context
    monkeypatch.setattr(module, "_setup", allocated)
    monkeypatch.setattr(module, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("validation failure after training")))
    folder = Path(run_config["paths"]["runs_dir"]) / run_config["run_id"]
    with pytest.raises(RuntimeError):
        module.run(run_config, folder)
    monkeypatch.setattr(module, "evaluate", evaluate)
    module.run(run_config, folder, resume=folder / "last.pt")
    first, second = folder / "attempts/000001", folder / "attempts/000002"
    assert json.loads((first / "environment.json").read_text())["allocation"]["job_id"] == 1
    assert json.loads((second / "environment.json").read_text())["allocation"]["job_id"] == 2
    outcome = json.loads((first / "outcome.json").read_text())
    assert outcome["status"] == "failed" and outcome["last_committed_epoch"] == 0
    assert outcome["observed_training"]["optimizer_updates"] == 4
    assert json.loads((second / "start_boundary.json").read_text())["completed_epoch"] == 0


def test_corrupt_resume_is_recorded_as_failure_without_masking_load_error(run_config, monkeypatch):
    module = install_cpu_edges(monkeypatch)
    folder = Path(run_config["paths"]["runs_dir"]) / run_config["run_id"]
    calls = iter([False, True])
    module.run(run_config, folder, stop_requested=lambda: next(calls))
    (folder / "last.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(Exception):
        module.run(run_config, folder, resume=folder / "last.pt")
    assert json.loads((folder / "run_summary.json").read_text())["status"] == "failed"
    events = json.loads((folder / "events.json").read_text())
    assert events[-1]["event"] == "failure" and events[-1]["last_committed_epoch"] is None
