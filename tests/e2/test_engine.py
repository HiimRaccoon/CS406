import copy
import pytest
import torch
from torch.utils.data import DataLoader

class Tiny(torch.nn.Module):
    def __init__(self,dropout=0.):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(dropout),torch.nn.Linear(8,124))
    def forward(self,views):
        x=torch.cat((views['global'][:,:,0,0],views['local'][:,:,0,0,0].mean(1,keepdim=True)),1)
        return self.net(x)

def tiny_data(n=7):
    rows=[]
    for i in range(n):
        base=torch.tensor([.1,.2,.3,.4])+i*.01
        rows.append(({'global':base[:3,None,None].expand(3,384,896),'local':base[3].expand(4,3,320,320)},i%3,f'synthetic/{i}.jpg'))
    return rows

def test_one_image_tail_matches_direct_sample_mean_steps():
    from src.e2.engine import train_epoch
    torch.manual_seed(22);a=Tiny();b=copy.deepcopy(a)
    oa,ob=[torch.optim.SGD(m.parameters(),lr=.01) for m in (a,b)]
    loader=DataLoader(tiny_data(3),batch_size=2)
    total=0
    for views,y,paths in loader:
        ob.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(b(views),y,reduction='sum');total+=loss.item()
        (loss/len(y)).backward();ob.step()
    result=train_epoch(a,loader,oa,None,None,device=torch.device('cpu'),runtime={'dtype':'float32'},accumulation=1,clipping={'enabled':False})
    assert result['samples']==3 and result['optimizer_updates']==result['microbatches']==2
    assert result['loss']==pytest.approx(total/3)
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)

def test_evaluate_preserves_manifest_paths_and_same_logits():
    from src.e2.engine import evaluate
    m=Tiny();loader=DataLoader(tiny_data(3),batch_size=2)
    result=evaluate(m,loader,device=torch.device('cpu'),runtime={'dtype':'float32'},n_train={i:1 for i in range(124)},labels=[str(i) for i in range(124)],collect_predictions=True)
    assert result['samples']==3 and [r['relative_path'] for r in result['predictions']]==[f'synthetic/{i}.jpg' for i in range(3)]
    assert all(len(r['top5'])==5 for r in result['predictions'])

@pytest.mark.parametrize('bad', ['amp','scaler','clipping','accumulation','nonfinite'])
def test_invalid_runtime_and_numerics_fail_without_update(bad):
    from src.e2.engine import train_epoch
    m=Tiny();opt=torch.optim.SGD(m.parameters(),lr=.01);runtime={'dtype':'float32'}
    if bad=='amp':runtime['amp']={'enabled':True}
    if bad=='nonfinite':
        with torch.no_grad():next(m.parameters()).fill_(float('nan'))
    with pytest.raises((ValueError,RuntimeError)):
        train_epoch(m,DataLoader(tiny_data(1),batch_size=2),opt,None,object() if bad=='scaler' else None,device=torch.device('cpu'),runtime=runtime,accumulation=2 if bad=='accumulation' else 1,clipping={'enabled':bad=='clipping'})
    assert not opt.state


def test_scheduler_independent_oracle_and_full_horizon(recipe):
    from src.e2.engine import UpdateScheduler
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
    from src.e2.engine import select_checkpoint
    best = {"epoch": 1, "macro_f1_supported": .5, "nll": 2.}
    for score in [.49, .5, .5000005]:
        assert not select_checkpoint({"epoch": 2, "macro_f1_supported": score, "nll": .1}, best)
    assert select_checkpoint({"epoch": 2, "macro_f1_supported": .500002}, best)
    with pytest.raises(ValueError):
        select_checkpoint({"epoch": 2, "macro_f1_supported": float("nan")}, best)
