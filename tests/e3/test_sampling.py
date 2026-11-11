import copy
import pytest
import torch


def test_natural_full_pass_and_restored_suffix():
    from src.e3.sampling import DrawSampler
    rows = [{"label_id": i % 2, "relative_path": str(i)} for i in range(5)]
    sampler = DrawSampler(rows, policy="natural", root_seed=1001, source_seed=2001, stage="pilot")
    iterator = iter(sampler); first = [next(iterator), next(iterator)]
    state = sampler.state_dict(); tail = list(iterator)
    assert sorted(first + tail) == [0,1,2,3,4]
    resumed = DrawSampler(rows, policy="natural", root_seed=1001, source_seed=2001, stage="pilot")
    resumed.load_state_dict(state)
    assert list(resumed) == tail
    resumed.set_window(1)
    assert sorted(resumed) == [0,1,2,3,4]
    bad = copy.deepcopy(state); bad["permutation"][0] = bad["permutation"][1]
    with pytest.raises(ValueError):
        resumed.load_state_dict(bad)


def test_crt_class_then_uniform_image_and_repeats(monkeypatch):
    from src.e3.sampling import DrawSampler
    rows = [{"label_id": i, "relative_path": str(i)} for i in range(124)]
    rows.append({"label_id": 0, "relative_path": "second-zero"})
    sampler = DrawSampler(rows, policy="crt", root_seed=1001, source_seed=2001, stage="pilot")
    draws = iter([0, 1, 0, 1, 123, 0])
    monkeypatch.setattr(torch, "randint", lambda *a, **k: torch.tensor([next(draws)]))
    iterator = iter(sampler)
    assert [next(iterator), next(iterator), next(iterator)] == [124, 124, 123]


def test_crt_restore_and_stream_independence():
    from src.e3.sampling import DrawSampler
    rows = [{"label_id": i, "relative_path": str(i)} for i in range(124)]
    sampler = DrawSampler(rows, policy="crt", root_seed=1001, source_seed=2001, stage="pilot")
    iterator = iter(sampler); [next(iterator) for _ in range(9)]
    state = sampler.state_dict(); expected = list(iterator)
    torch.rand(100)
    restored = DrawSampler(rows, policy="crt", root_seed=1001, source_seed=2001, stage="pilot")
    restored.load_state_dict(state)
    assert list(restored) == expected
    bad = copy.deepcopy(state); bad["source_seed"] = 3001
    with pytest.raises(ValueError):
        restored.load_state_dict(bad)
