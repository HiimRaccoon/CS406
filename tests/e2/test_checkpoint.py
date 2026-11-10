import copy
import random
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from .test_engine import Tiny,tiny_data

def components(run_config):
    from src.e2.engine import UpdateScheduler
    c=copy.deepcopy(run_config);m=Tiny(.2);o=torch.optim.AdamW(m.parameters(),lr=1e-4)
    s=UpdateScheduler(o,c['recipe']['scheduler'],139992)
    return c,m,o,s,{'train':DataLoader(tiny_data(3),batch_size=2,generator=torch.Generator().manual_seed(99))}

def test_discard_restore_and_replay_matches_uninterrupted_state(run_config):
    from src.e2.checkpoint import capture_state,restore_state
    from src.e2.engine import train_epoch
    c,m,o,s,ls=components(run_config)
    args=dict(model=m,optimizer=o,scheduler=s,scaler=None,loaders=ls)
    progress={'completed_epoch':0,'updates':0,'history':[],'selected':None,'selected_weights':None}
    committed=capture_state(progress=progress,**args)
    def epoch(model,opt,sched,loaders):
        return train_epoch(model,loaders['train'],opt,sched,None,device=torch.device('cpu'),runtime={'dtype':'float32'},accumulation=1,clipping={'enabled':False})
    epoch(m,o,s,ls);expected=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();draws=(random.random(),np.random.rand())
    _,other,oo,ss,ll=components(run_config)
    restored=restore_state(committed,model=other,optimizer=oo,scheduler=ss,scaler=None,loaders=ll)
    assert restored==progress
    epoch(other,oo,ss,ll)
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in expected.items())
    assert torch.equal(rng,torch.get_rng_state()) and draws==(random.random(),np.random.rand())
    assert s.state_dict()==ss.state_dict()
    for k,values in o.state_dict()['state'].items():
        for name,value in values.items():assert torch.equal(value,oo.state_dict()['state'][k][name])
    assert torch.equal(ls['train'].generator.get_state(),ll['train'].generator.get_state())

def test_official_e2_boundary_and_topology_are_required(run_config,tmp_path):
    from src.e2.checkpoint import save_checkpoint
    c,m,o,s,ls=components(run_config)
    args=dict(model=m,optimizer=o,scheduler=s,scaler=None,loaders=ls,identity={'scientific_runtime':{'evidence_kind':'official_uit_slurm'},'audit':{}},progress={'completed_epoch':0,'updates':0,'history':[],'selected':None,'selected_weights':None})
    for mutation in [lambda x:x.update(experiment='E1-HR'),lambda x:x['views'].update(local_count=3),lambda x:x['recipe']['budget'].update(planned_max_updates=32)]:
        bad=copy.deepcopy(c);mutation(bad)
        with pytest.raises(ValueError):save_checkpoint(tmp_path/'last.pt',config=bad,**args)

def test_atomic_failure_retains_previous_last(run_config,tmp_path,monkeypatch):
    from src.e2.checkpoint import save_checkpoint,read_checkpoint,restore_checkpoint
    c,m,o,s,ls=components(run_config);identity={'scientific_runtime':{'evidence_kind':'official_uit_slurm'},'audit':{}}
    args=dict(model=m,optimizer=o,scheduler=s,scaler=None,loaders=ls,config=c,identity=identity,progress={'completed_epoch':0,'updates':0,'history':[],'selected':None,'selected_weights':None})
    path=tmp_path/'last.pt';save_checkpoint(path,**args);assert path.exists()
    before=path.read_bytes();monkeypatch.setattr('src.e2.checkpoint.os.replace',lambda *a:(_ for _ in ()).throw(OSError('injected write failure')))
    with pytest.raises(OSError):save_checkpoint(path,**args)
    assert path.read_bytes()==before
    state=read_checkpoint(path,config=c,identity=identity)
    assert state['kind']=='official_e2_checkpoint_v1' and state['views']==c['views']
    with pytest.raises(ValueError):restore_checkpoint(tmp_path/'best.pt',**{k:v for k,v in args.items() if k!='progress'})
