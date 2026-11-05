"""Prepared E2-T11 probe: opt-in, separate target permission, zero official credit."""
from __future__ import annotations
import copy
import datetime as dt
import gc
import math
import os
import time
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from src.e2.checkpoint import _atomic,atomic_json,capture_state,restore_state
from src.e2.config import (CONTRACT_SHA256,ROOT,load_recipe,read_json,require_approval,
                           sha256_file,topology,validate_acceptance_request)
from src.e2.data import E2Dataset,E2Loader,EpochSampler,FixedManifest,derived_seed,seed_worker
from src.e2.engine import train_epoch
from src.e2.runtime import inspect_target_inputs,prepare


def mechanics_config(recipe):
    return {'kind':'e2_target_checkpoint_fixture','contract_sha256':CONTRACT_SHA256,
            **topology(recipe),'recipe':copy.deepcopy(recipe),'official_run':False,
            'boundary_unit':'three-image train-only mechanics segment; never an official epoch',
            'official_epoch_credit':0,'official_update_credit':0}


def g2_candidate(identity,device,evidence_path):
    return {'gate':'E2-G2','approved':False,'contract_sha256':CONTRACT_SHA256,
            **topology(load_recipe()),'identity':identity,'device':device,
            'acceptance_evidence':str(evidence_path),
            'acceptance_evidence_sha256':sha256_file(Path(evidence_path)),
            'status':'candidate_only_requires_explicit_user_E2_G2_review'}


def require_target_work():
    value=os.environ.get('CS406_E2_TARGET_AUTH')
    if not value:raise ValueError('Separate E2 target-work authorization required before GPU access')
    record=require_approval(Path(value),'E2-TARGET-AUTH')
    if record.get('authorized_tasks')!=['E2-T11'] or record.get('target_gpu_authorized') is not True:
        raise ValueError('Explicit bounded E2-T11 target acceptance permission required')
    return {'path':value,'sha256':sha256_file(Path(value))}


class TrainTriple(Dataset):
    def __init__(self,dataset,indices):self.dataset,self.indices=dataset,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        epoch,i=index
        return self.dataset[epoch,self.indices[i]]


def triple_loader(manifest,config,indices):
    recipe,seed=config['recipe'],config['seed']
    dataset=TrainTriple(E2Dataset(manifest,config['paths']['data_root'],'train',seed,
                                 recipe['preprocessing']['augmentation']),indices)
    return E2Loader(dataset,sampler=EpochSampler(dataset,seed),batch_size=2,drop_last=False,
                    num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(derived_seed(seed,'loader','train')))


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left:
            assert_same(left[k], right[k])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def state_args(context):
    return {k:context[k] for k in ('model','optimizer','scheduler','scaler','loaders')}


def save_fixture(path,context,progress,fixture):
    state=capture_state(**state_args(context),progress=progress)
    state.update(kind='e2_target_checkpoint_fixture_v1',fixture=fixture,
                 contract_sha256=CONTRACT_SHA256,official_run=False)
    _atomic(path,lambda p:torch.save(state,p))


def restore_fixture(path,context,fixture):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if (state.get('kind')!='e2_target_checkpoint_fixture_v1' or state.get('fixture')!=fixture
            or state.get('official_run') is not False or state.get('contract_sha256')!=CONTRACT_SHA256):
        raise ValueError('Different nonofficial acceptance fixture')
    return restore_state(state,**state_args(context))


def segment(context,recipe,epoch):
    """Observe actual two-call order, full/tail shapes and all view gradients."""
    context['loaders']['train'].sampler.set_epoch(epoch)
    calls=[];inputs=[];fused=[];outputs=[]
    def retain(module,args):
        views=args[0]
        for v in views.values():
            v.requires_grad_(True);v.retain_grad()
        inputs.append(views)
    def encoder(module,args):calls.append(list(args[0].shape))
    def classifier(module,args):fused.append(list(args[0].shape))
    def result(module,args,output):
        assert output.dtype==torch.float32 and torch.isfinite(output).all()
        outputs.append(list(output.shape))
    model=context['model']
    hooks=[model.register_forward_pre_hook(retain),model.features.register_forward_pre_hook(encoder),
           model.classifier.register_forward_pre_hook(classifier),model.register_forward_hook(result)]
    try:
        observed=train_epoch(model,context['loaders']['train'],context['optimizer'],context['scheduler'],None,
                             device=context['device'],runtime=recipe['runtime'],accumulation=1,
                             clipping=recipe['optimizer']['gradient_clipping'])
    finally:
        for hook in hooks:hook.remove()
    assert calls==[[2,3,384,896],[8,3,320,320],[1,3,384,896],[4,3,320,320]]
    assert fused==[[2,1536],[1,1536]] and outputs==[[2,124],[1,124]]
    for views in inputs:
        for name,v in views.items():
            assert v.grad is not None and torch.isfinite(v.grad).all()
            grad=v.grad.flatten(0,1) if name=='local' else v.grad
            assert (grad.flatten(1).abs().sum(1)>0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert observed['samples']==3 and observed['optimizer_updates']==2 and math.isfinite(observed['loss'])
    numerical={k:observed[k] for k in ('loss','samples','optimizer_updates','microbatches','lr_first','lr_last')}
    return numerical,{'measurements':observed,'encoder_calls':calls,'fused':fused,'logits':outputs,
                      'all_global_and_local_input_gradients_finite_nonzero':True}


def _progress(context,history,selected_weights=None):
    return {'completed_epoch':len(history),'updates':2*len(history),'history':copy.deepcopy(history),
            'selected':{'epoch':1,'fixture_sentinel':True,'macro_f1_supported':0.0},
            'selected_weights':cpu_copy(context['model'].state_dict()) if selected_weights is None else cpu_copy(selected_weights)}


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('CS406_E2_GPU_SMOKE')!='1',reason='E2 target acceptance is separately gated')
def test_uit_real_train_boundary_resume():
    authorization=require_target_work()  # Before target/device/image access.
    request=validate_acceptance_request(read_json(Path(os.environ['CS406_E2_ACCEPTANCE_REQUEST'])))
    if Path(request['target_authorization']).resolve()!=Path(authorization['path']).resolve():
        raise ValueError('Acceptance request must bind this target-work receipt')
    output=Path(os.environ['CS406_E2_SMOKE_DIR']).resolve()
    evidence_root=Path(request['paths']['evidence_root']).resolve()
    runs=Path(request['paths']['runs_dir']).resolve()
    if (not output.is_relative_to(evidence_root) or output==evidence_root or output.is_relative_to(runs)
            or output.is_relative_to(ROOT.resolve())):
        raise ValueError('Use a fresh nonofficial evidence attempt outside the clean checkout and run root')
    output.mkdir(parents=True,exist_ok=False)
    recipe=load_recipe();fixture=mechanics_config(recipe)
    evidence={'kind':'e2_target_acceptance_v1',**topology(recipe),'contract_sha256':CONTRACT_SHA256,
              'stage':'acceptance','status':'running','official_run':False,'test_images_opened':0,
              'validation_images_decoded':0,'target_authorization':authorization,
              'request':request,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'official_epochs_completed':0,'official_update_credit':0,'G2_approved':False}
    path=output/'evidence.json';atomic_json(path,evidence)
    started=time.perf_counter()
    try:
        checked=inspect_target_inputs(request)
        manifest=FixedManifest();records=manifest.records('train')
        indices=[next(i for i,r in enumerate(records) if (r.width,r.height)==layout)
                 for layout in ((1536,662),(1030,703))]
        indices.append(next(i for i in range(len(records)) if i not in indices))
        evidence.update(identity=checked,train_members=[{'relative_path':records[i].relative_path,
                        'sha256':records[i].sha256,'raw_wh':[records[i].width,records[i].height]} for i in indices])
        def fresh():
            ctx=prepare(request,checked)
            ctx['loaders']['train']=triple_loader(manifest,request,indices)
            assert ctx['model_metadata']['parameters']==28010716
            assert ctx['model_metadata']['classifier_parameters']==190588
            assert ctx['scheduler'].total_updates==139992
            return ctx
        context=fresh();evidence['model_metadata']=context['model_metadata']
        first,first_audit=segment(context,recipe,0)
        progress=_progress(context,[first]);last=output/'fixture_last.pt'
        save_fixture(last,context,progress,fixture);committed=sha256_file(last)
        second,second_audit=segment(context,recipe,1)
        expected_progress=_progress(context,[first,second],progress['selected_weights'])
        expected=cpu_copy(capture_state(**state_args(context),progress=expected_progress))
        del context;gc.collect();torch.cuda.empty_cache()
        context=fresh();restored=restore_fixture(last,context,fixture)
        assert_same(restored,progress);assert context['scheduler'].completed==2
        replay,replay_audit=segment(context,recipe,1)
        actual_progress=_progress(context,[first,replay],restored['selected_weights'])
        assert_same(cpu_copy(capture_state(**state_args(context),progress=actual_progress)),expected)
        assert context['scheduler'].completed==4 and sha256_file(last)==committed
        evidence.update(status='pass',segments={'first':first_audit,'uncommitted_reference':second_audit,'replay':replay_audit},
                        resume={'model_optimizer_full_scheduler_rng_all_cuda_loader_history_selection_bitwise_equal':True,
                                'fixture_updates_after_replay':4,'full_horizon':139992,'fixture_last_sha256':committed,
                                'method':'discard uncommitted segment; reconstruct; restore completed boundary; replay',
                                'scope':'bounded deterministic-stack reconstruction, not full-natural-epoch or OS-kill proof'},
                        parameter_count_checks={'classifier':190588,'total':28010716},
                        timing={'scope':'whole bounded acceptance attempt including audit/setup/reconstruction',
                                'elapsed_seconds':time.perf_counter()-started},
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(path,evidence)
        atomic_json(output/'g2_candidate.json',g2_candidate(checked,checked['scientific_runtime']['device_and_driver'],path))
        inventory={p.name:{'bytes':p.stat().st_size,'sha256':sha256_file(p),'role':'nonofficial_acceptance_output'}
                   for p in output.iterdir() if p.is_file() and p.name!='output_inventory.json'}
        atomic_json(output/'output_inventory.json',inventory)
    except BaseException as exc:
        evidence.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(path,evidence)
        raise
