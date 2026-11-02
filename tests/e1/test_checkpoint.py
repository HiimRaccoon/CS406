import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from .test_engine import Tiny, tiny_data


def components(run_config):
    from src.e1.engine import UpdateScheduler
    c = copy.deepcopy(run_config)
    c["recipe"]["batch"]["updates_per_epoch"] = 4
    c["recipe"]["budget"]["planned_max_updates"] = 32
    model = Tiny(.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = UpdateScheduler(optimizer, {"warmup_updates": 2, "warmup_start_factor": .1, "cosine_floor_lr": 1e-6}, 32)
    loader = DataLoader(tiny_data(), batch_size=2, generator=torch.Generator().manual_seed(99))
    return c, model, optimizer, scheduler, {"train": loader}


def test_cpu_full_state_boundary_zero_resume_and_atomic_recovery(run_config, tmp_path, monkeypatch):
    from src.e1.checkpoint import save_checkpoint, restore_checkpoint, read_checkpoint
    from src.e1.engine import train_epoch
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    config, model, opt, sched, loaders = components(run_config)
    identity = {"evidence_kind": "synthetic_cpu_fixture"}
    path = tmp_path / "last.pt"
    kwargs = dict(model=model, optimizer=opt, scheduler=sched, scaler=None, config=config, identity=identity, loaders=loaders)
    zero = {"completed_epoch": 0, "updates": 0, "history": [], "selected": None, "selected_weights": None}
    save_checkpoint(path, progress=zero, **kwargs)
    assert restore_checkpoint(path, **kwargs)["completed_epoch"] == 0
    def epoch(m, o, s, ls):
        return train_epoch(m, ls["train"], o, s, None, device=torch.device("cpu"),
                           runtime={"dtype": "float32"}, accumulation=1, clipping={"enabled": False})
    train = epoch(model, opt, sched, loaders)
    progress = {"completed_epoch": 1, "updates": 4, "history": [{"epoch": 1, "train": train}],
                "selected": {"epoch": 1, "macro_f1_supported": .5}, "selected_weights": copy.deepcopy(model.state_dict())}
    save_checkpoint(path, progress=progress, **kwargs)
    epoch(model, opt, sched, loaders)
    expected = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), torch.get_rng_state(), sched.state_dict()
    draws = (random.random(), np.random.rand())
    _, other, oo, ss, ll = components(run_config)
    restore_checkpoint(path, model=other, optimizer=oo, scheduler=ss, scaler=None, config=config, identity=identity, loaders=ll)
    epoch(other, oo, ss, ll)
    for k, v in expected[0].items():
        assert torch.equal(v, other.state_dict()[k])
    assert torch.equal(expected[2], torch.get_rng_state()) and expected[3] == ss.state_dict()
    for key, value in expected[1]["state"].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, oo.state_dict()["state"][key][field])
    assert draws == (random.random(), np.random.rand())
    restore_checkpoint(path, **kwargs)
    before = path.read_bytes()
    monkeypatch.setattr("src.e1.checkpoint.os.replace", lambda *a: (_ for _ in ()).throw(OSError("injected atomic failure")))
    with pytest.raises(OSError):
        save_checkpoint(path, progress=progress, **kwargs)
    assert path.read_bytes() == before
    assert read_checkpoint(path, config=config, identity=identity)["progress"]["updates"] == 4


def test_wrong_identity_budget_pilot_and_best_resume_rejected(run_config, tmp_path):
    from src.e1.checkpoint import save_checkpoint, restore_checkpoint, read_checkpoint
    config, model, opt, sched, loaders = components(run_config)
    identity = {"evidence_kind": "synthetic_cpu_fixture"}
    kwargs = dict(model=model, optimizer=opt, scheduler=sched, scaler=None, config=config, identity=identity, loaders=loaders)
    p = tmp_path / "last.pt"
    progress = {"completed_epoch": 0, "updates": 0, "history": [], "selected": None, "selected_weights": None}
    save_checkpoint(p, progress=progress, **kwargs)
    with pytest.raises(ValueError):
        read_checkpoint(p, config=config, identity={"wrong": True})
    bad = copy.deepcopy(config)
    bad["recipe"]["budget"]["max_epochs"] = 9
    with pytest.raises(ValueError):
        read_checkpoint(p, config=bad, identity=identity)
    state = torch.load(p, weights_only=False)
    state["role"] = "best"
    best = tmp_path / "best.pt"
    torch.save(state, best)
    with pytest.raises(ValueError):
        restore_checkpoint(best, **kwargs)
    state["kind"] = "bounded_pilot_checkpoint"
    torch.save(state, p)
    with pytest.raises(ValueError):
        read_checkpoint(p, config=config, identity=identity)
