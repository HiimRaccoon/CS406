import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.pilot.checkpoint import load_checkpoint, save_checkpoint
from src.pilot.engine import WarmupCosine, is_better, train_window


class TinyPilot(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(dropout), torch.nn.Linear(8, 124))

    def forward(self, views):
        return self.net(views["global"])


RUNTIME = {"amp": {"enabled": False}, "scaler": {"enabled": False}}


def epoch(model, loader, opt, scaler, scheduler=None, accumulation=1):
    return train_window(model, loader, opt, scaler, device=torch.device("cpu"), runtime=RUNTIME,
                        accumulation=accumulation, clipping={"enabled": False}, scheduler=scheduler)


def data(n=7):
    return [({"global": torch.arange(4).float() * .1 + i * .01}, i % 3) for i in range(n)]


def test_accumulation_short_tail_matches_original_image_weighting():
    torch.manual_seed(31)
    a = TinyPilot()
    b = copy.deepcopy(a)
    oa, ob = torch.optim.SGD(a.parameters(), lr=.01), torch.optim.SGD(b.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ra = epoch(a, DataLoader(data(), batch_size=2), oa, scaler, accumulation=3)
    rb = epoch(b, DataLoader(data(), batch_size=6), ob, scaler)
    assert ra["samples"] == rb["samples"] == 7
    assert ra["optimizer_updates"] == rb["optimizer_updates"] == 2
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_epoch_boundary_resume_rng_schedule_and_budget(tmp_path):
    torch.manual_seed(29)
    random.seed(29)
    np.random.seed(29)
    model = TinyPilot(.2)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    sched = WarmupCosine(opt, total_updates=8, warmup_updates=2, start_factor=.1, min_lr=.00001)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(9)
    loader = DataLoader(data(), batch_size=2, generator=generator)
    epoch(model, loader, opt, scaler, sched)
    cfg, identity = {"seed": 29}, {"synthetic": True}
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                    cfg=cfg, epoch=1, updates=4, history=[{"epoch": 1}], best=None,
                    loader_generators={"train": generator}, identity=identity)
    epoch(model, loader, opt, scaler, sched)
    expected_weights = copy.deepcopy(model.state_dict())
    expected_rng = (random.random(), np.random.rand(), torch.rand(3))
    restored = TinyPilot(.2)
    other_opt = torch.optim.AdamW(restored.parameters(), lr=.001)
    other_sched = WarmupCosine(other_opt, 8, 2, .1, .00001)
    other_scaler = torch.amp.GradScaler("cuda", enabled=False)
    other_generator = torch.Generator()
    kwargs = dict(model=restored, optimizer=other_opt, scheduler=other_sched, scaler=other_scaler,
                  cfg=cfg, loader_generators={"train": other_generator}, identity=identity)
    state = load_checkpoint(path, **kwargs)
    assert state["epoch"] == 1 and other_sched.completed == 4
    result = epoch(restored, DataLoader(data(), batch_size=2, generator=other_generator), other_opt, other_scaler, other_sched)
    assert result["optimizer_updates"] == 4 and other_sched.completed == 8
    assert other_opt.param_groups[0]["lr"] == pytest.approx(.00001)
    for key, value in restored.state_dict().items():
        assert torch.equal(value, expected_weights[key])
    assert random.random() == expected_rng[0]
    assert np.random.rand() == expected_rng[1]
    assert torch.equal(torch.rand(3), expected_rng[2])
    with pytest.raises(ValueError, match="config/budget/seed"):
        load_checkpoint(path, **{**kwargs, "cfg": {"seed": 30}})


def test_scheduler_formula_and_checkpoint_ties():
    opt = torch.optim.SGD(TinyPilot().parameters(), lr=1.)
    sched = WarmupCosine(opt, 6, 2, .1, .01)
    assert opt.param_groups[0]["lr"] == pytest.approx(.1)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(.55)
    sched.step()
    assert opt.param_groups[0]["lr"] == 1.
    best = {"macro_f1_supported": .5, "nll": 1.}
    near = {"macro_f1_supported": .50000001, "nll": .9}
    assert not is_better(near, best, .0001, "earlier")
    assert is_better(near, best, .0001, "nll")
