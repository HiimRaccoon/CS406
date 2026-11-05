"""Fabricated trusted parser envelopes; temporary files, never UIT evidence."""
import copy
import csv
from pathlib import Path

import pytest
import torch


@pytest.fixture
def completed_trials(recipe,tmp_path):
    from src.e2.checkpoint import atomic_json,save_checkpoint
    from src.e2.config import ROOT,CONTRACT_SHA256,resolve_run,sha256_file,fingerprint
    from src.e2.engine import UpdateScheduler
    from src.e2.metrics import Metrics124
    from torch.utils.data import DataLoader
    from .test_engine import Tiny,tiny_data
    paths={'data_root':str(tmp_path/'simulated-data'),'weights':str(tmp_path/'convnext_tiny-983f1562.pth'),
           'runs_dir':str(tmp_path/'simulated-runs'),'acceptance_record':str(tmp_path/'g2.json'),
           'authorization_record':str(tmp_path/'g3a.json'),'evidence_root':str(tmp_path/'evidence'),'conformance_report':str(tmp_path/'conformance.json')}
    atomic_json(tmp_path/'fixture_manifest.json',{'kind':'synthetic_artifact_parser_fixture','real_gpu_work':False})
    atomic_json(Path(paths['conformance_report']),{'simulation_only':True})
    sources={p:'a'*64 for p in recipe['identity_and_evidence']['source_fingerprint_closure']}
    d=recipe['deployment']
    actual={'cpu_threads':2,'matmul_precision':'highest','tf32_matmul':False,'tf32_cudnn':False,'cudnn_benchmark':False,
            'cudnn_deterministic':True,'deterministic_algorithms':True,'warn_only':False}
    science={'consumed_source_files':sources,'consumed_source_sha256':fingerprint(sources),'experiment':'E2','views':recipe['views'],'representation_dimension':1536,'fusion':'global_then_mean_local','view_execution':'global_then_flattened_locals_batch_major',
        'contract_sha256':CONTRACT_SHA256,'software':recipe['runtime']['software_lock'],'effective_runtime':actual,'runtime_policy':recipe['runtime'],
        'device_and_driver':{'name':d['expected_gpu_class'],'driver':'fixture','total_bytes':34896609280},
        'reviewed_allocation_policy':{k:d[k] for k in ('partition','account','qos','gres','ntasks','cpus_per_task','slurm_mem_mib','job_walltime')},
        'dataset_root':paths['data_root'],'fingerprints':{k:v['sha256'] for k,v in recipe['data']['artifacts'].items()},
        'preprocessing_config_sha256':recipe['preprocessing']['config_sha256'],'preprocessing_implementation_sha256':recipe['preprocessing']['implementation_sha256'],
        'pretrained_sha256':recipe['model']['pretrained']['sha256'],'accepted_conformance_sha256':sha256_file(Path(paths['conformance_report'])),
        'hr_preprocessing_conformance':{'input_chw':[3,576,1344],'test_images_opened':0},'evidence_kind':'official_uit_slurm'}
    identity={'scientific_runtime':science,'audit':{'clean_git_commit':'a'*40,'git_clean':True}}
    evidence=tmp_path/'accepted_evidence.json'
    atomic_json(evidence,{'kind':'e2_target_acceptance_v1','status':'pass','official_run':False,'experiment':'E2',
                         'views':recipe['views'],'representation_dimension':1536,'fusion':'global_then_mean_local','view_execution':'global_then_flattened_locals_batch_major','contract_sha256':CONTRACT_SHA256,'test_images_opened':0,'identity':identity})
    atomic_json(Path(paths['acceptance_record']),{'gate':'E2-G2','approved':True,'experiment':'E2','views':recipe['views'],'representation_dimension':1536,'fusion':'global_then_mean_local','view_execution':'global_then_flattened_locals_batch_major',
          'contract_sha256':CONTRACT_SHA256,'identity':identity,'acceptance_evidence':str(evidence),'acceptance_evidence_sha256':sha256_file(evidence)})
    configs=[resolve_run(recipe,stage='tuning',lr=lr,seed=1001,paths=paths) for lr in recipe['campaign']['lr_candidates']]
    atomic_json(Path(paths['authorization_record']),{'gate':'E2-G3a','approved':True,'experiment':'E2','views':recipe['views'],'representation_dimension':1536,'fusion':'global_then_mean_local','view_execution':'global_then_flattened_locals_batch_major',
          'contract_sha256':CONTRACT_SHA256,'acceptance_sha256':sha256_file(Path(paths['acceptance_record'])),'run_ids':[c['run_id'] for c in configs],'declarations':{c['run_id']:c['config_sha256'] for c in configs}})
    with (ROOT/'artifacts/baseline124/class_support.csv').open(encoding='utf-8',newline='') as stream: classes=list(csv.DictReader(stream))
    targets=torch.tensor([int(r['label_id']) for r in classes for _ in range(int(r['val']))])
    logits=torch.zeros(len(targets),124);logits[torch.arange(len(targets)),targets]=100
    metric=Metrics124();metric.update(logits,targets,0.)
    metrics=metric.compute({int(r['label_id']):int(r['train']) for r in classes},[r['label'] for r in classes])
    folders=[]
    for config in configs:
        folder=Path(paths['runs_dir'])/config['run_id'];folder.mkdir(parents=True)
        atomic_json(folder/'config_resolved.json',config)
        model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'])
        scheduler=UpdateScheduler(optimizer,recipe['scheduler'],139992)
        state=scheduler.state_dict();state['completed']=139992;scheduler.load_state_dict(state)
        history=[{'epoch':e,'train':{'samples':34997,'optimizer_updates':17499,'microbatches':17499},'validation':copy.deepcopy(metrics)} for e in range(1,9)]
        selected={'epoch':1,'macro_f1_supported':1.,'nll':0.}
        progress={'completed_epoch':8,'updates':139992,'history':history,'selected':selected,'selected_weights':copy.deepcopy(model.state_dict())}
        args=dict(model=model,optimizer=optimizer,scheduler=scheduler,scaler=None,config=config,progress=progress,identity=identity,
                  loaders={'train':DataLoader(tiny_data(),generator=torch.Generator().manual_seed(99))})
        save_checkpoint(folder/'last.pt',**args);save_checkpoint(folder/'best.pt',**args)
        atomic_json(folder/'history.json',history)
        attempt=folder/'attempts/000001'
        atomic_json(attempt/'request.json',{'run_id':config['run_id'],'config_sha256':config['config_sha256'],'contract_sha256':CONTRACT_SHA256,'operation':'created'})
        atomic_json(attempt/'environment.json',{'audit':identity['audit'],'actual_torch_settings':actual})
        atomic_json(attempt/'start_boundary.json',{'completed_epoch':0,'updates':0})
        atomic_json(attempt/'outcome.json',{'status':'complete','last_committed_epoch':8,'last_committed_updates':139992})
        atomic_json(folder/'events.json',[{'event':'created','attempt_id':'000001'}, {'event':'attempt_environment','attempt_id':'000001','environment_path':'attempts/000001/environment.json','environment_sha256':sha256_file(attempt/'environment.json')},
                    *[{'event':'epoch_committed','attempt_id':'000001','epoch':e,'updates':17499*e} for e in range(1,9)],{'event':'complete','attempt_id':'000001'}])
        summary={'kind':'official_e2_run_summary_v1','experiment':'E2','views':recipe['views'],'representation_dimension':1536,'fusion':'global_then_mean_local','view_execution':'global_then_flattened_locals_batch_major','budget':recipe['budget'],
             'status':'complete','evidence_kind':'official_uit_slurm','contract_sha256':CONTRACT_SHA256,'test_images_opened':0,'identity':identity,
             **{k:config[k] for k in ('run_id','stage','seed','lr','config_sha256')},'planned_epochs':8,'completed_epochs':8,'validations':8,
             'optimizer_updates':139992,'planned_optimizer_updates':139992,'termination_reason':'eight_epoch_budget_complete','selected':selected,
             'checkpoint_sha256':{n:sha256_file(folder/n) for n in ('best.pt','last.pt')},'history_sha256':sha256_file(folder/'history.json')}
        atomic_json(folder/'run_summary.json',summary);folders.append(folder)
    return folders


def test_full_parser_and_locked_selection(recipe,completed_trials,tmp_path):
    from src.e2.campaign import validate_completed_run,select_lr,write_selection,read_selection
    for folder in completed_trials: assert validate_completed_run(folder)['summary']['completed_epochs']==8
    selected=select_lr(recipe,[{'run_dir':str(p)} for p in completed_trials])
    assert selected['lr']==2.5e-5 and selected['next_stage']=='STOP_FOR_USER_E2_G3b_REVIEW'
    p=tmp_path/'selection.json';write_selection(p,recipe,selected)
    assert read_selection(p,recipe)==selected


def test_rehashed_short_history_and_wrong_selected_weights_rejected(completed_trials):
    from src.e2.campaign import validate_completed_run
    from src.e2.config import read_json,sha256_file
    from src.e2.checkpoint import atomic_json
    folder=completed_trials[0];history=read_json(folder/'history.json')
    atomic_json(folder/'history.json',history[:7]);summary=read_json(folder/'run_summary.json')
    summary['history_sha256']=sha256_file(folder/'history.json');atomic_json(folder/'run_summary.json',summary)
    with pytest.raises(ValueError):validate_completed_run(folder)
    folder=completed_trials[1];state=torch.load(folder/'best.pt',weights_only=False);key=next(iter(state['model']))
    state['model'][key]=state['model'][key]+1;torch.save(state,folder/'best.pt')
    summary=read_json(folder/'run_summary.json');summary['checkpoint_sha256']['best.pt']=sha256_file(folder/'best.pt');atomic_json(folder/'run_summary.json',summary)
    with pytest.raises(ValueError):validate_completed_run(folder)


def test_attempt_evidence_mutation_rejected(completed_trials):
    from src.e2.campaign import validate_completed_run
    p=completed_trials[0]/'attempts/000001/environment.json';p.write_text('{}')
    with pytest.raises(ValueError):validate_completed_run(completed_trials[0])


def test_docs_only_commit_audit_is_not_campaign_identity(completed_trials):
    from src.e2.campaign import validate_completed_run
    from src.e2.checkpoint import atomic_json
    from src.e2.config import read_json
    folder=completed_trials[0];summary=read_json(folder/'run_summary.json')
    summary['identity']['audit']['clean_git_commit']='b'*40;atomic_json(folder/'run_summary.json',summary)
    assert validate_completed_run(folder)['summary']['completed_epochs']==8


@pytest.mark.parametrize('missing',['environment.json','start_boundary.json'])
def test_completed_attempt_requires_initialized_records(completed_trials,missing):
    from src.e2.campaign import validate_completed_run
    (completed_trials[0]/'attempts/000001'/missing).unlink()
    with pytest.raises(ValueError):validate_completed_run(completed_trials[0])


def test_attempt_event_cannot_reference_missing_directory(completed_trials):
    from src.e2.campaign import validate_completed_run
    from src.e2.checkpoint import atomic_json
    from src.e2.config import read_json
    folder=completed_trials[0];events=read_json(folder/'events.json')
    events.insert(0,{'event':'created','attempt_id':'000000'})
    atomic_json(folder/'events.json',events)
    with pytest.raises(ValueError):validate_completed_run(folder)


@pytest.mark.parametrize('status',['failed_preflight','abandoned_request','abandoned_outcome'])
def test_prior_uninitialized_or_interrupted_attempt_is_auditable(completed_trials,status):
    from src.e2.campaign import validate_completed_run
    from src.e2.checkpoint import atomic_json
    from src.e2.config import read_json,sha256_file
    from src.e2.train import read_attempt_metadata
    folder=completed_trials[0];config=read_json(folder/'config_resolved.json')
    earlier=folder/'attempts/000000';earlier.mkdir()
    binding={k:config[k] for k in ('run_id','config_sha256','contract_sha256')}
    atomic_json(earlier/'request.json',binding)
    prefix=[]
    if status=='failed_preflight':
        atomic_json(earlier/'outcome.json',{'status':'failed','last_committed_epoch':None,'last_committed_updates':None})
        prefix=[{'event':'created','attempt_id':'000000'},{'event':'failure','attempt_id':'000000'}]
    else:
        if status=='abandoned_request':
            (earlier/'request.json').write_text('{')
        else:
            atomic_json(earlier/'environment.json',{'observed_allocation':'synthetic'})
            atomic_json(earlier/'start_boundary.json',{'completed_epoch':0,'updates':0})
            (earlier/'outcome.json').write_text('{')
            prefix=[{'event':'created','attempt_id':'000000'},
                    {'event':'attempt_environment','attempt_id':'000000','environment_path':'attempts/000000/environment.json',
                     'environment_sha256':sha256_file(earlier/'environment.json')},
                    {'event':'attempt_boundary','attempt_id':'000000','completed_epoch':0,'updates':0}]
        records={name:read_attempt_metadata(earlier/name) for name in
                 ('request.json','environment.json','start_boundary.json','outcome.json')}
        issues={name:record[1] for name,record in records.items() if record[1] is not None}
        atomic_json(earlier/'abandoned.json',{**binding,'status':'terminated_without_outcome','metadata_issues':issues})
        prefix.append({'event':'attempt_abandoned','attempt_id':'000000','metadata_issues':issues})
    atomic_json(folder/'events.json',prefix+read_json(folder/'events.json'))
    assert validate_completed_run(folder)['summary']['completed_epochs']==8
    if status=='abandoned_request':
        (earlier/'request.json').write_text('{"altered":')
        with pytest.raises(ValueError):validate_completed_run(folder)
