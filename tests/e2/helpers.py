"""Test-owned nonofficial envelopes around production state primitives."""
from pathlib import Path
import copy
import torch
from src.e2.config import fingerprint,scientific_identity
from src.e2.checkpoint import capture_state,restore_state,_atomic

def save_fixture(path, *, config,identity,progress,**components):
    state=capture_state(progress=progress,**components)
    state.update(kind='e2_test_checkpoint_fixture_v1',role=Path(path).stem,config=copy.deepcopy(config),
                 config_sha256=fingerprint(config),identity=copy.deepcopy(identity),views=config['views'],experiment='E2')
    if Path(path).stem=='best':state['model']=copy.deepcopy(progress['selected_weights'])
    _atomic(path,lambda p:torch.save(state,p))

def read_fixture(path, *, config,identity):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if (state['kind']!='e2_test_checkpoint_fixture_v1' or state['config_sha256']!=fingerprint(config)
            or fingerprint(state['config'])!=fingerprint(config)
            or scientific_identity(state['identity'])!=scientific_identity(identity)):
        raise ValueError('Nonofficial fixture identity/kind mismatch')
    return state

def restore_fixture(path, *, config,identity,**components):
    if Path(path).name!='last.pt':raise ValueError('Fixture restore role mismatch')
    return restore_state(read_fixture(path,config=config,identity=identity),**components)
