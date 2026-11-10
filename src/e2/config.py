"""Frozen E2 inputs and immutable, allocation-free run declarations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/E2/e2_numeric_contract_v1.json"
CONTRACT_SHA256 = "ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72"
G1_PATH = ROOT / "docs/E2/e2_g1_approval.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError(f"Duplicate JSON key: {k}")
            obj[k] = v
        return obj
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def _frozen(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("Expected exact frozen E2 contract SHA-256")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text().split() != [CONTRACT_SHA256, path.name]:
        raise ValueError("Missing or inconsistent contract sidecar")
    return read_json(path)


TOPOLOGY = {'global_wh':[896,384], 'local_wh':[320,320], 'local_count':4,
            'local_order':['left','middle_left','middle_right','right_raised'],
            'fusion':'global_then_mean_local', 'representation_dimension':1536,
            'view_execution':'global_then_flattened_locals_batch_major'}


def require_approval(path, gate, run_id=None, *, contract_sha256=CONTRACT_SHA256, bindings=None):
    record = read_json(path)
    if (gate not in {'E2-G1','E2-G2','E2-G3a','E2-G3b','E2-TARGET-AUTH'}
            or contract_sha256 != CONTRACT_SHA256 or record.get('gate') != gate
            or record.get('approved') is not True or record.get('experiment') != 'E2'
            or record.get('contract_sha256') != CONTRACT_SHA256
            or not isinstance(record.get('views'),dict)
            or any(record['views'].get(k) != v for k,v in TOPOLOGY.items())):
        raise ValueError(f'Missing reviewed {gate} approval for E2 topology/contract')
    if run_id is not None and run_id not in record.get('run_ids',[]):
        raise ValueError('Run is not authorized by '+gate)
    if bindings and any(record.get(k) != v for k,v in bindings.items()):
        raise ValueError('Approval binding mismatch')
    return record


def load_recipe(path=CONTRACT_PATH):
    recipe = _frozen(path)
    receipt = require_approval(G1_PATH,'E2-G1')
    if (receipt.get('execution_method') != 'native_task_by_task'
            or receipt.get('authorized_tasks') != [f'E2-T{i:02}' for i in range(1,11)]):
        raise ValueError('Named G1 implementation scope required')
    return recipe


def validate_recipe(recipe):
    try:
        if fingerprint(recipe) != fingerprint(load_recipe()):
            raise ValueError('Recipe differs from the exact G1-frozen E2 contract')
    except (TypeError,OverflowError) as exc:
        raise ValueError('Invalid E2 recipe') from exc
    return copy.deepcopy(recipe)


def topology(recipe):
    return {'experiment':'E2','views':copy.deepcopy(recipe['views']),
            **{k:TOPOLOGY[k] for k in ('representation_dimension','fusion','view_execution')}}


def scientific_run_payload(config):
    return {k:copy.deepcopy(config[k]) for k in
            ('experiment','views','representation_dimension','fusion','view_execution',
             'stage','seed','lr','budget','contract_sha256')}


def _paths(paths,final=False):
    required={'data_root','weights','runs_dir','evidence_root','conformance_report','acceptance_record','authorization_record'}
    if final:required.add('selection')
    if (not isinstance(paths,dict) or set(paths) != required
            or any(not isinstance(v,str) or not v.strip() for v in paths.values())):
        raise ValueError('Explicit E2 paths required; unknown overrides rejected')
    return copy.deepcopy(paths)


def resolve_run(recipe, *, stage, lr, seed, paths):
    recipe=validate_recipe(recipe)
    allowed=[1001] if stage=='tuning' else [2001,3001,4001] if stage=='final' else []
    if type(seed) is not int or seed not in allowed:
        raise ValueError('Invalid stage or seed role')
    if type(lr) not in {int,float} or not math.isfinite(lr) or lr not in recipe['campaign']['lr_candidates']:
        raise ValueError('Reviewed LR grid required')
    paths=_paths(paths,stage=='final')
    result={'kind':'official_e2_run_config_v1','recipe':recipe,**topology(recipe),
            'stage':stage,'lr':lr,'seed':seed,'budget':copy.deepcopy(recipe['budget']),
            'contract_sha256':CONTRACT_SHA256,'run_id':f'e2-{stage}-lr{lr:.8f}-seed{seed}','paths':paths}
    result['scientific_sha256']=fingerprint(scientific_run_payload(result))
    if stage=='final':result['selection_sha256']=sha256_file(Path(paths['selection']))
    result['config_sha256']=fingerprint(result)
    return result


def validate_run(config):
    try:
        expected=resolve_run(config['recipe'],stage=config['stage'],lr=config['lr'],seed=config['seed'],paths=config['paths'])
        if fingerprint(config) != fingerprint(expected):
            raise ValueError('Changed or unknown E2 declaration fields')
    except (KeyError,TypeError) as exc:
        raise ValueError('Incomplete E2 declaration') from exc
    return copy.deepcopy(config)


def acceptance_request(recipe, *, paths, target_authorization):
    recipe=validate_recipe(recipe)
    if not isinstance(target_authorization,str) or not target_authorization.strip():
        raise ValueError('Explicit target-work receipt path required')
    request={'kind':'e2_target_acceptance_request_v1','recipe':recipe,**topology(recipe),
             'stage':'acceptance','seed':1001,'lr':2.5e-5,'budget':copy.deepcopy(recipe['budget']),
             'contract_sha256':CONTRACT_SHA256,'paths':_paths(paths),
             'target_authorization':target_authorization,'official_run':False}
    request['request_sha256']=fingerprint(request)
    return request


def validate_acceptance_request(request):
    try:
        expected=acceptance_request(request['recipe'],paths=request['paths'],target_authorization=request['target_authorization'])
        if fingerprint(request) != fingerprint(expected):raise ValueError('Changed acceptance request')
    except (KeyError,TypeError) as exc:
        raise ValueError('Incomplete acceptance request') from exc
    return copy.deepcopy(request)


def runtime_identity(identity):
    if (not isinstance(identity,dict) or set(identity) != {'scientific_runtime','audit'}
            or not isinstance(identity['scientific_runtime'],dict) or not identity['scientific_runtime']
            or not isinstance(identity['audit'],dict)):
        raise ValueError('Scientific runtime and separate audit identity required')
    fingerprint(identity['scientific_runtime'])
    return copy.deepcopy(identity['scientific_runtime'])


scientific_identity = runtime_identity
