"""Four-trial HR declarations, complete evidence validation and locked selection."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from .checkpoint import read_checkpoint
from .config import (CONTRACT_SHA256,fingerprint,load_recipe,read_json,resolve_run,sha256_file,
                     scientific_identity,validate_recipe,validate_run)
from .engine import select_checkpoint


def plan_tuning(recipe: dict, paths: dict) -> list[dict]:
    recipe=validate_recipe(recipe)
    return [resolve_run(recipe,stage='tuning',lr=lr,seed=1001,paths=paths) for lr in recipe['campaign']['lr_candidates']]


def _write_new(path: Path, value) -> None:
    data=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    with Path(path).open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def write_declarations(output: Path, declarations: list[dict]) -> None:
    declarations=[validate_run(c) for c in declarations]
    if not declarations or len({c['run_id'] for c in declarations})!=len(declarations):
        raise ValueError('Unique HR declarations required')
    stage=declarations[0]['stage']
    if any(c['stage']!=stage for c in declarations) or len(declarations)!=(4 if stage=='tuning' else 3):
        raise ValueError('Exactly four tuning or three final declarations required')
    if stage=='tuning' and {c['lr'] for c in declarations}!={2.5e-5,5e-5,1e-4,2e-4}:
        raise ValueError('Wrong HR candidate set')
    if stage=='final' and ({c['seed'] for c in declarations}!={2001,3001,4001} or len({c['lr'] for c in declarations})!=1):
        raise ValueError('Wrong final seed/LR set')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for c in declarations:_write_new(output/(c['run_id']+'.json'),c)
    _write_new(output/'ledger.json',{'kind':'e1hr_campaign_declarations_v1','experiment':'E1-HR','resolution_wh':[1344,576],
        'contract_sha256':CONTRACT_SHA256,'stage':stage,'status':'declared_not_executed','execution_authorized':False,
        'runs':[{'run_id':c['run_id'],'config_sha256':c['config_sha256']} for c in declarations]})


def _science(config, identity):
    science=scientific_identity(identity);recipe=config['recipe'];pre=recipe['preprocessing']
    sources=science['consumed_source_files']
    if (set(sources)!=set(recipe['identity_and_evidence']['source_fingerprint_closure'])
            or science['consumed_source_sha256']!=fingerprint(sources)
            or science['experiment']!='E1-HR' or science['resolution_wh']!=[1344,576]
            or science['contract_sha256']!=CONTRACT_SHA256 or science['evidence_kind']!='official_uit_slurm'
            or science['software']!=recipe['runtime']['software_lock'] or science['runtime_policy']!=recipe['runtime']
            or science['dataset_root']!=config['paths']['data_root']
            or science['fingerprints']!={k:v['sha256'] for k,v in recipe['data']['artifacts'].items()}
            or science['preprocessing_config_sha256']!=pre['config_sha256']
            or science['preprocessing_implementation_sha256']!=pre['implementation_sha256']
            or science['pretrained_sha256']!=recipe['model']['pretrained']['sha256']
            or science['device_and_driver']['name']!=recipe['deployment']['expected_gpu_class']):
        raise ValueError('Invalid accepted HR scientific/runtime identity')
    return science


def _attempt_inventory(folder: Path, config: dict, events: list[dict]) -> dict:
    from .train import read_attempt_metadata
    attempts=folder/'attempts'
    if not attempts.is_dir():raise ValueError('Missing HR attempt records')
    inventory={}
    roots=sorted(p for p in attempts.iterdir() if p.is_dir())
    if not roots or any(not p.name.isdigit() for p in roots):raise ValueError('Invalid attempt identities')
    if {e.get('attempt_id') for e in events}!={p.name for p in roots}:
        raise ValueError('Attempt directories and event lineage mismatch')
    for p in roots:
        records={name:read_attempt_metadata(p/name) for name in
                 ('request.json','environment.json','start_boundary.json','outcome.json')}
        request,request_issue=records['request.json']
        outcome,outcome_issue=records['outcome.json']
        own_events=[e for e in events if e['attempt_id']==p.name]
        names={e['event'] for e in own_events}
        abandoned=None
        if outcome_issue is not None:
            abandoned=read_json(p/'abandoned.json')
            issues={name:result[1] for name,result in records.items() if result[1] is not None}
            if (abandoned.get('status')!='terminated_without_outcome' or abandoned.get('metadata_issues')!=issues
                    or any(abandoned.get(k)!=config[k] for k in ('run_id','config_sha256','contract_sha256'))
                    or not any(e.get('event')=='attempt_abandoned' and e.get('metadata_issues')==issues for e in own_events)):
                raise ValueError('Interrupted attempt metadata was not explicitly reconciled')
        elif outcome.get('status') not in {'complete','interrupted','failed'}:
            raise ValueError('Invalid attempt outcome')
        if request_issue is not None:
            if abandoned is None or names & {'created','resume','recover_initialization','attempt_environment','attempt_boundary','epoch_started','epoch_committed'}:
                raise ValueError('Missing initialized attempt request')
        elif any(request.get(k)!=config[k] for k in ('run_id','config_sha256','contract_sha256')):
            raise ValueError('Attempt declaration mismatch')
        initialized=(outcome is not None and outcome.get('status') in {'complete','interrupted'})
        training_events={'attempt_boundary','epoch_started','training_observed','epoch_committed','complete','interrupted'}
        environment,env_issue=records['environment.json']
        boundary,boundary_issue=records['start_boundary.json']
        if (initialized or names & (training_events|{'attempt_environment'})) and env_issue is not None:
            raise ValueError('Missing initialized attempt environment')
        if (initialized or names & training_events) and boundary_issue is not None:
            raise ValueError('Missing initialized attempt start boundary')
        if boundary_issue is None:
            if (type(boundary['completed_epoch']) is not int or not 0<=boundary['completed_epoch']<=8
                    or boundary['updates']!=boundary['completed_epoch']*17499):
                raise ValueError('Invalid committed attempt boundary')
        if env_issue is None:
            rel=(p/'environment.json').relative_to(folder).as_posix()
            if not any(e.get('environment_path')==rel and e.get('environment_sha256')==sha256_file(p/'environment.json') for e in own_events):
                raise ValueError('Attempt environment hash/lineage mismatch')
        for event in own_events:
            if 'environment_path' in event and (event['environment_path']!=f'attempts/{p.name}/environment.json'
                    or env_issue is not None or event.get('environment_sha256')!=sha256_file(p/'environment.json')):
                raise ValueError('Dangling attempt environment evidence')
        for f in p.iterdir():
            if not f.is_file() or not f.resolve().is_relative_to(folder):raise ValueError('Unsafe attempt file')
            inventory[f.relative_to(folder).as_posix()]=sha256_file(f)
    final=read_json(roots[-1]/'outcome.json')
    if final.get('status')!='complete' or final.get('last_committed_epoch')!=8 or final.get('last_committed_updates')!=139992:
        raise ValueError('Final attempt did not close original full budget')
    return inventory


def validate_completed_run(run_dir: Path) -> dict:
    from .train import approved_identity
    folder=Path(run_dir).resolve()
    try:
        summary=read_json(folder/'run_summary.json')
        if (summary.get('kind')!='official_e1hr_run_summary_v1' or summary.get('status')!='complete'
                or summary.get('evidence_kind')!='official_uit_slurm' or summary.get('test_images_opened')!=0
                or summary.get('contract_sha256')!=CONTRACT_SHA256):
            raise ValueError('Only complete official HR evidence is eligible')
        config=validate_run(read_json(folder/'config_resolved.json'));recipe=config['recipe'];identity=summary['identity']
        if (folder.name!=config['run_id'] or folder!=(Path(config['paths']['runs_dir'])/config['run_id']).resolve()
                or any(summary[k]!=config[k] for k in ('run_id','stage','seed','lr','config_sha256','experiment','resolution_wh','budget'))
                or summary['planned_epochs']!=8 or summary['completed_epochs']!=8 or summary['validations']!=8
                or summary['optimizer_updates']!=139992 or summary['planned_optimizer_updates']!=139992
                or summary['termination_reason']!='eight_epoch_budget_complete'):
            raise ValueError('Wrong HR identity or incomplete original budget')
        _science(config,identity)
        if scientific_identity(approved_identity(config))!=scientific_identity(identity):
            raise ValueError('Completed HR source/runtime differs from accepted identity')
        for name in ('best.pt','last.pt'):
            if sha256_file(folder/name)!=summary['checkpoint_sha256'][name]:raise ValueError('Checkpoint hash mismatch')
        if sha256_file(folder/'history.json')!=summary['history_sha256']:raise ValueError('History hash mismatch')
        history=read_json(folder/'history.json')
        if len(history)!=8:raise ValueError('Missing full history')
        selected=None
        for epoch,row in enumerate(history,1):
            train,val=row['train'],row['validation']
            if (row['epoch']!=epoch or train['samples']!=34997 or train['optimizer_updates']!=17499 or train['microbatches']!=17499
                    or val['samples']!=7497 or val['supported_classes']!=106 or val['output_classes']!=124
                    or any(type(val[k]) not in {int,float} or not math.isfinite(val[k]) or not 0<=val[k]<=1
                           for k in ('top1','top3','top5','macro_f1_supported','weighted_f1'))):
                raise ValueError('Invalid/incomplete epoch evidence')
            candidate={'epoch':epoch,'macro_f1_supported':val['macro_f1_supported'],'nll':val['nll']}
            if select_checkpoint(candidate,selected):selected=candidate
        if selected!=summary['selected']:raise ValueError('Wrong selected checkpoint metric/epoch')
        last=read_checkpoint(folder/'last.pt',config=config,identity=identity)
        best=read_checkpoint(folder/'best.pt',config=config,identity=identity)
        if (last['role']!='last' or best['role']!='best' or last['progress']['completed_epoch']!=8
                or last['progress']['history']!=history or best['progress']['selected']!=selected or last['progress']['selected']!=selected
                or set(best['model'])!=set(last['progress']['selected_weights'])
                or any(not torch.equal(v,last['progress']['selected_weights'][k]) for k,v in best['model'].items())):
            raise ValueError('Committed history/selected weights lineage mismatch')
        events=read_json(folder/'events.json')
        if not isinstance(events,list) or not events or events[-1].get('event')!='complete':raise ValueError('Missing terminal event')
        inventory=_attempt_inventory(folder,config,events)
        return {'run_dir':str(folder),'config':config,'summary':summary,'summary_sha256':sha256_file(folder/'run_summary.json'),
                'best_sha256':summary['checkpoint_sha256']['best.pt'],'events_sha256':sha256_file(folder/'events.json'),
                'attempts_sha256':fingerprint(inventory)}
    except (KeyError,TypeError,FileNotFoundError,IndexError) as exc:
        raise ValueError('Incomplete official HR evidence') from exc


def select_lr(recipe: dict, trial_records: list[dict]) -> dict:
    recipe=validate_recipe(recipe)
    if len(trial_records)!=4 or any(set(r)!={'run_dir'} for r in trial_records):raise ValueError('Four run directories required, no caller scores')
    rows=sorted([validate_completed_run(Path(r['run_dir'])) for r in trial_records],key=lambda r:r['summary']['lr'])
    if (len({r['run_dir'] for r in rows})!=4 or {r['summary']['lr'] for r in rows}!=set(recipe['campaign']['lr_candidates'])
            or any(r['summary']['stage']!='tuning' or r['summary']['seed']!=1001 for r in rows)):
        raise ValueError('Missing/duplicate/wrong-stage HR candidate; stop for review')
    if any(scientific_identity(r['summary']['identity'])!=scientific_identity(rows[0]['summary']['identity']) for r in rows):
        raise ValueError('All four HR trials require the same accepted scientific/runtime identity')
    maximum=max(r['summary']['selected']['macro_f1_supported'] for r in rows)
    lr=min(r['summary']['lr'] for r in rows if maximum-r['summary']['selected']['macro_f1_supported']<=1e-6)
    result={'kind':'e1hr_lr_selection_v1','experiment':'E1-HR','resolution_wh':[1344,576],'stage':'tuning','seed':1001,
        'budget':recipe['budget'],'contract_sha256':CONTRACT_SHA256,'lr':lr,'tuning_seed':1001,'maximum_supported_f1':maximum,
        'tolerance':1e-6,'tie_break':'lowest_base_lr_within_global_max_tolerance',
        'trials':[{k:r[k] for k in ('run_dir','summary_sha256','best_sha256','events_sha256','attempts_sha256')} for r in rows],
        'next_stage':'STOP_FOR_USER_HR_G3b_REVIEW','final_execution_authorized':False}
    result['selection_fingerprint']=fingerprint(result)
    return result


def _verify_selection(recipe,selection):
    try:
        expected=select_lr(recipe,[{'run_dir':r['run_dir']} for r in selection['trials']])
        if fingerprint(expected)!=fingerprint(selection):raise ValueError('Modified or stale locked HR selection')
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete HR selection') from exc


def read_selection(path: Path, recipe: dict) -> dict:
    value=read_json(path);_verify_selection(recipe,value);return value


def write_selection(path: Path, recipe: dict, selection: dict) -> None:
    _verify_selection(recipe,selection);_write_new(path,selection)


def plan_final(recipe: dict, selection: dict, paths: dict) -> list[dict]:
    recipe=validate_recipe(recipe)
    if 'selection' not in paths or read_selection(Path(paths['selection']),recipe)!=selection:
        raise ValueError('Exact persisted locked HR selection required')
    return [resolve_run(recipe,stage='final',lr=selection['lr'],seed=seed,paths=paths) for seed in (2001,3001,4001)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('declare-tuning','select-lr','declare-final'))
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--selection',type=Path)
    args=parser.parse_args();recipe=load_recipe();inputs=read_json(args.input)
    if args.action=='declare-tuning':write_declarations(args.output,plan_tuning(recipe,inputs))
    elif args.action=='select-lr':
        write_selection(args.output,recipe,select_lr(recipe,inputs));print('STOP for explicit HR-G3b review')
    else:
        if args.selection is None:parser.error('--selection required')
        inputs['selection']=str(args.selection);write_declarations(args.output,plan_final(recipe,read_selection(args.selection,recipe),inputs))
    print('No jobs submitted or training authorized')


if __name__=='__main__':main()
