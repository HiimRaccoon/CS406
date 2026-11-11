import copy
import math
import pytest
import torch
from torch import nn
from torch.nn import functional as F


def test_tail_accumulation_matches_direct_sample_mean_updates():
    from src.e3.engine import _train_batches
    torch.manual_seed(7)
    actual = nn.Linear(2,124); expected = copy.deepcopy(actual)
    x=torch.arange(10,dtype=torch.float32).reshape(5,2)/10; y=torch.arange(5)
    batches=[(x[:2],y[:2]),(x[2:4],y[2:4]),(x[4:],y[4:])]
    aopt=torch.optim.SGD(actual.parameters(),lr=.1); eopt=torch.optim.SGD(expected.parameters(),lr=.1)
    class Schedule:
        completed=0
        def set_lr(self): return .1
        def step(self): self.completed+=1
    schedule=Schedule()
    for start,end in [(0,4),(4,5)]:
        eopt.zero_grad(); F.cross_entropy(expected(x[start:end]),y[start:end]).backward()
        torch.nn.utils.clip_grad_norm_(expected.parameters(),1.,error_if_nonfinite=True)
        eopt.step()
    result=_train_batches(actual,batches,aopt,schedule,accumulation=2,clipping={"max_norm":1.,"norm_type":2.,"error_if_nonfinite":True,"foreach":False},device="cpu")
    assert result["presentations"] == 5 and result["updates"] == 2
    for a,b in zip(actual.parameters(),expected.parameters()): assert torch.allclose(a,b,atol=1e-7,rtol=1e-6)


def test_schedule_endpoints_and_exhaustion():
    from src.e3.config import policy
    from src.e3.engine import make_head_scheduler
    opt=torch.optim.SGD(nn.Linear(768,124).parameters(),lr=1.)
    s=make_head_scheduler(opt,protocol=policy()["candidate_profiles"][0])
    expected={0:.001/69,68:.001,69:.001,1379:.00001}
    for u in range(1380):
        lr=s.set_lr()
        if u in expected: assert lr == pytest.approx(expected[u])
        if u==724:
            assert lr == pytest.approx(.001*(.01+.99*(1+math.cos(math.pi*(724-69)/1310))/2))
        s.step()
    with pytest.raises(ValueError): s.set_lr()


def test_chronological_selection_and_nonfinite_failure():
    from src.e3.engine import select_head,_train_batches
    assert select_head({"macro_f1_supported":.5},None)
    assert not select_head({"macro_f1_supported":.5000005},{"macro_f1_supported":.5})
    assert select_head({"macro_f1_supported":.500002},{"macro_f1_supported":.5})
    head=nn.Linear(2,124); opt=torch.optim.SGD(head.parameters(),lr=.1)
    class Schedule:
        completed=0
        def set_lr(self): pass
        def step(self): self.completed+=1
    s=Schedule(); before=copy.deepcopy(head.state_dict())
    with pytest.raises(ValueError):
        _train_batches(head,[(torch.full((1,2),float('nan')),torch.tensor([0]))],opt,s,accumulation=1,
                       clipping={"max_norm":1.,"norm_type":2.,"error_if_nonfinite":True,"foreach":False},device="cpu")
    assert s.completed==0
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in before.items())
