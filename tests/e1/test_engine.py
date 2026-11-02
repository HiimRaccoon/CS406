import copy

import pytest
import torch
from torch.utils.data import DataLoader


class Tiny(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(dropout), torch.nn.Linear(8, 124))
    def forward(self, x):
        return self.net(x)


def tiny_data(n=7):
    return [(torch.tensor([.1, .2, .3, .4]) + i*.01, i % 3, f"synthetic/{i}.jpg") for i in range(n)]


def test_sample_weighted_short_accumulation_matches_direct_batches():
    from src.e1.engine import train_epoch
    torch.manual_seed(22)
    a = Tiny()
    b = copy.deepcopy(a)
    oa, ob = [torch.optim.SGD(m.parameters(), lr=.01) for m in (a, b)]
    kwargs = dict(device=torch.device("cpu"), runtime={"dtype": "float32"}, clipping={"enabled": False})
    ra = train_epoch(a, DataLoader(tiny_data(), batch_size=2), oa, None, None, accumulation=3, **kwargs)
    rb = train_epoch(b, DataLoader(tiny_data(), batch_size=6), ob, None, None, accumulation=1, **kwargs)
    assert ra["samples"] == rb["samples"] == 7
    assert ra["optimizer_updates"] == rb["optimizer_updates"] == 2
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_scheduler_independent_oracle_and_full_horizon(recipe):
    from src.e1.engine import UpdateScheduler
    opt = torch.optim.SGD(Tiny().parameters(), lr=1.)
    schedule = {"warmup_updates": 2, "warmup_start_factor": .1, "cosine_floor_lr": .01}
    s = UpdateScheduler(opt, schedule, 6)
    used = []
    for _ in range(6):
        used.append(opt.param_groups[0]["lr"])
        s.step()
    assert used == pytest.approx([.1, .55, 1., .7525, .2575, .01])
    assert used[-1] == .01
    with pytest.raises(ValueError):
        s.step()
    official = UpdateScheduler(torch.optim.SGD(Tiny().parameters(), lr=1e-4), recipe["scheduler"], 139992)
    state = official.state_dict()
    state["completed"] = 139991
    official.load_state_dict(state)
    assert official.optimizer.param_groups[0]["lr"] == 1e-6
    with pytest.raises(ValueError):
        official.load_state_dict({**state, "total_updates": 52497})


def test_chronological_tau_earlier_comparator_ignores_nll():
    from src.e1.engine import select_checkpoint
    best = {"epoch": 1, "macro_f1_supported": .5, "nll": 2.}
    for score in [.49, .5, .5000005]:
        assert not select_checkpoint({"epoch": 2, "macro_f1_supported": score, "nll": .1}, best)
    assert select_checkpoint({"epoch": 2, "macro_f1_supported": .500002}, best)
    with pytest.raises(ValueError):
        select_checkpoint({"epoch": 2, "macro_f1_supported": float("nan")}, best)


def test_validation_exports_same_checkpoint_predictions_and_nonfinite_fails():
    from src.e1.engine import evaluate, train_epoch
    m = Tiny()
    result = evaluate(m, DataLoader(tiny_data(), batch_size=2), device=torch.device("cpu"),
                      runtime={"dtype": "float32"}, n_train={i: 1 for i in range(124)},
                      labels=[str(i) for i in range(124)], collect_predictions=True)
    assert result["samples"] == len(result["predictions"]) == 7
    assert len({r["relative_path"] for r in result["predictions"]}) == 7
    with torch.no_grad():
        next(m.parameters()).fill_(float("nan"))
    with pytest.raises(RuntimeError):
        train_epoch(m, DataLoader(tiny_data(), batch_size=2), torch.optim.SGD(m.parameters(), lr=.1),
                    None, None, device=torch.device("cpu"), runtime={"dtype": "float32"}, accumulation=1, clipping={"enabled": False})
