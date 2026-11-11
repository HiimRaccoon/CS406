import copy
import pytest
import torch


@pytest.mark.parametrize("family,count,dimension", [("E1",95356,768),("E2",190588,1536)])
def test_paired_initial_state_counts_isolation_and_lr_independence(family,count,dimension):
    from src.e3.config import policy
    from src.e3.model import make_paired_heads, head_state_sha256, build_head_optimizer
    profiles = policy()["candidate_profiles"]
    outer = torch.get_rng_state().clone()
    a,b,identity = make_paired_heads(family=family,seed=2001,protocol=profiles[0],stage="pilot",root_seed=1001)
    assert torch.equal(outer,torch.get_rng_state())
    assert sum(p.numel() for p in a.parameters()) == count
    assert torch.equal(a.weight,b.weight) and a.weight.data_ptr() != b.weight.data_ptr()
    assert head_state_sha256(a) == head_state_sha256(b) == identity["tensor_state_sha256"]
    assert a.weight.abs().max() <= 1/dimension**.5
    c,_,_ = make_paired_heads(family=family,seed=2001,protocol=profiles[1],stage="pilot",root_seed=1001)
    assert torch.equal(a.weight,c.weight) and torch.equal(a.bias,c.bias)
    optimizer, groups = build_head_optimizer(a,protocol=profiles[0])
    assert [g["weight_decay"] for g in optimizer.param_groups] == [.01,0.]
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(a.weight),id(a.bias)}
    assert len(groups["weight"]) == len(groups["bias"]) == 1


def test_rejects_foreign_head_and_protocol_overrides():
    from src.e3.config import policy
    from src.e3.model import make_paired_heads,build_head_optimizer
    p = copy.deepcopy(policy()["candidate_profiles"][0]); p["source_head_warm_start"] = True
    with pytest.raises(ValueError):
        make_paired_heads(family="E1",seed=2001,protocol=p,stage="pilot",root_seed=1001)
    with pytest.raises(ValueError):
        build_head_optimizer(torch.nn.Linear(1,124),protocol=policy()["candidate_profiles"][0])
