"""Atomic official boundaries; best is reporting-only, last is resume-only."""
from __future__ import annotations

import copy
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint, scientific_identity, validate_run, topology


def _atomic(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                # Windows may briefly deny replacement while another process has
                # a handle open. Retry this same atomic write, never training work.
                if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                    raise
                time.sleep(.05 * 2**attempt)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    def write(p):
        with p.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    _atomic(path, write)


def atomic_text(path: Path, text: str) -> None:
    def write(p):
        with p.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    _atomic(path, write)


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def capture_state(*, model, optimizer, scheduler, scaler, loaders, progress):
    if scaler is not None:raise ValueError('E2 scaler policy is inactive')
    return copy.deepcopy({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'scaler':None,'rng':_rng(),
        'loader_rng':{k:v.generator.get_state() for k,v in loaders.items()},'progress':progress})


def restore_state(state, *, model, optimizer, scheduler, scaler, loaders):
    required={'model','optimizer','scheduler','scaler','rng','loader_rng','progress'}
    if not required <= state.keys() or scaler is not None or state['scaler'] is not None:
        raise ValueError('Incomplete full state or changed scaler policy')
    if set(state['loader_rng']) != set(loaders):raise ValueError('Loader state mismatch')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    for name,loader in loaders.items():
        loader.generator.set_state(state['loader_rng'][name])
        if hasattr(loader,'prepare_resume'):loader.prepare_resume(state['progress']['completed_epoch'])
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['cpu'])
    if state['rng']['cuda'] is not None:
        if not torch.cuda.is_available():raise ValueError('Cannot restore CUDA state on CPU')
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return copy.deepcopy(state['progress'])


def _progress(progress, config, scheduler):
    e=progress['completed_epoch'];u=progress['updates'];history=progress['history']
    if (type(e) is not int or not 0 <= e <= 8 or type(u) is not int or u != e*17499
            or scheduler['total_updates'] != 139992 or scheduler['completed'] != u
            or len(history) != e or [r['epoch'] for r in history] != list(range(1,e+1))):
        raise ValueError('Invalid official E2 completed-epoch boundary or original budget')
    if e==0:
        if progress['selected'] is not None or progress['selected_weights'] is not None:
            raise ValueError('Boundary zero cannot have selected weights')
        return
    from .engine import select_checkpoint
    selected=None
    for row in history:
        train,val=row['train'],row['validation']
        if (train['samples'] != 34997 or train['optimizer_updates'] != 17499
                or val['samples'] != 7497 or val['supported_classes'] != 106 or val['output_classes'] != 124):
            raise ValueError('Official completed boundary requires full train and validation coverage')
        candidate={'epoch':row['epoch'],'macro_f1_supported':val['macro_f1_supported'],'nll':val['nll']}
        if select_checkpoint(candidate,selected):selected=candidate
    if selected != progress['selected'] or not progress['selected_weights']:
        raise ValueError('Selected state differs from chronological validation history')


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, config, progress, identity, loaders):
    config=validate_run(config);scientific_identity(identity);path=Path(path)
    if path.name not in {'best.pt','last.pt'}:raise ValueError('Official best.pt/last.pt role required')
    _progress(progress,config,scheduler.state_dict())
    if path.name=='best.pt' and progress['selected'] is None:raise ValueError('Unselected best checkpoint')
    state=capture_state(model=model,optimizer=optimizer,scheduler=scheduler,scaler=scaler,loaders=loaders,progress=progress)
    state.update(kind='official_e2_checkpoint_v1',role=path.stem,config=config,
                 config_sha256=fingerprint(config),identity=identity,**topology(config['recipe']))
    if path.stem=='best':state['model']=copy.deepcopy(progress['selected_weights'])
    _atomic(path,lambda p:torch.save(state,p))


def read_checkpoint(path, *, config, identity):
    config=validate_run(config)
    state=torch.load(path,map_location='cpu',weights_only=False)
    required={'kind','role','config','config_sha256','identity','model','optimizer','scheduler','scaler','rng','loader_rng','progress',*topology(config['recipe'])}
    try:
        if (set(state) != required or state['kind'] != 'official_e2_checkpoint_v1'
                or state['role'] not in {'best','last'} or state['config_sha256'] != fingerprint(config)
                or fingerprint(state['config']) != fingerprint(config)
                or scientific_identity(state['identity']) != scientific_identity(identity)
                or any(state[k] != v for k,v in topology(config['recipe']).items()) or state['scaler'] is not None):
            raise ValueError('E2 checkpoint identity/config/kind mismatch')
        _progress(state['progress'],config,state['scheduler'])
        if state['role']=='best' and state['progress']['selected'] is None:raise ValueError('Unselected best state')
    except (KeyError,TypeError) as exc:raise ValueError('Incomplete official E2 checkpoint') from exc
    return state


def restore_checkpoint(path, *, model, optimizer, scheduler, scaler, config, identity, loaders):
    if Path(path).name != 'last.pt':raise ValueError('Resume only from same-run last.pt, never best.pt')
    state=read_checkpoint(path,config=config,identity=identity)
    if state['role'] != 'last':raise ValueError('Wrong resume role')
    return restore_state(state,model=model,optimizer=optimizer,scheduler=scheduler,scaler=scaler,loaders=loaders)
