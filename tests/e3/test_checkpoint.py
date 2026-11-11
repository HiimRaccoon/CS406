import copy
import random
import numpy as np
import pytest
import torch


@pytest.mark.parametrize("arm",["natural","crt"])
def test_capture_discard_restore_preserves_all_rng_sampler_and_optimizer(arm):
    from src.e3.checkpoint import capture_head_state,restore_head_state
    from src.e3.sampling import DrawSampler
    from src.e3.engine import HeadSchedule
    head=torch.nn.Linear(768,124); opt=torch.optim.AdamW(head.parameters(),lr=.1)
    sched=HeadSchedule(opt,{"total_updates":8,"warmup_updates":1,"cosine_floor_factor":.01},.1)
    sampler=DrawSampler([{"relative_path":str(i),"label_id":i} for i in range(124)],policy=arm,
                        root_seed=1001,source_seed=2001,stage="pilot")
    class Loader:
        generator=torch.Generator().manual_seed(17)
    loaders={"train":Loader()}; progress={"window":0,"presentations":0,"updates":0,"history":[],"selected":None,"selected_weights":None}
    state=capture_head_state(head=head,optimizer=opt,scheduler=sched,loaders=loaders,sampler=sampler,progress=progress)
    expected=(random.random(),np.random.rand(),torch.rand(2),list(sampler))
    for p in head.parameters(): p.data.add_(3)
    restore_head_state(state,head=head,optimizer=opt,scheduler=sched,loaders=loaders,sampler=sampler)
    actual=(random.random(),np.random.rand(),torch.rand(2),list(sampler))
    assert actual[0]==expected[0] and actual[1]==expected[1] and torch.equal(actual[2],expected[2]) and actual[3]==expected[3]
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in state["head"].items())


def test_authoritative_last_roundtrip_identity_and_best_rejection(tmp_path):
    from src.e3.checkpoint import capture_head_state,save_head_checkpoint,read_head_checkpoint
    from src.e3.model import make_paired_heads,build_head_optimizer
    from src.e3.engine import make_head_scheduler
    from src.e3.config import policy
    from src.e3.sampling import DrawSampler
    from tests.e3.helpers import declaration
    config=declaration(); protocol=policy()["candidate_profiles"][0]
    head,_,_=make_paired_heads(family="E1",seed=2001,protocol=protocol,stage="pilot",root_seed=1001)
    opt,_=build_head_optimizer(head,protocol=protocol); sched=make_head_scheduler(opt,protocol=protocol)
    sampler=DrawSampler([{"relative_path":"x","label_id":0}],policy="natural",root_seed=1001,source_seed=2001,stage="pilot")
    progress={"window":0,"presentations":0,"updates":0,"history":[],"selected":None,"selected_weights":None}
    state=capture_head_state(head=head,optimizer=opt,scheduler=sched,loaders={},sampler=sampler,progress=progress)
    save_head_checkpoint(tmp_path/"last.pt",state=state,config=config,identity={"source":"same"},role="last")
    restored=read_head_checkpoint(tmp_path/"last.pt",config=config,identity={"source":"same"})
    assert restored["progress"]==progress
    with pytest.raises(ValueError): read_head_checkpoint(tmp_path/"last.pt",config=config,identity={"source":"other"})
    with pytest.raises(ValueError): read_head_checkpoint(tmp_path/"best.pt",config=config,identity={"source":"same"})
