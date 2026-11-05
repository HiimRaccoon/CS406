import copy
import json
import pytest

def test_e2_topology_and_common_controls_are_immutable(recipe,run_config):
    from src.e2.config import load_recipe,validate_run
    assert load_recipe()==recipe
    c=validate_run(run_config)
    assert c['experiment']=='E2' and c['views']['global_wh']==[896,384]
    assert c['views']['local_count']==4 and c['representation_dimension']==1536
    assert c['view_execution']=='global_then_flattened_locals_batch_major'
    assert c['budget']['planned_max_updates']==139992
    assert recipe['batch']['updates_per_epoch']==17499
    assert recipe['model']['parameter_count_oracles']['classifier']==190588
    assert recipe['model']['parameter_count_oracles']['total']==28010716

@pytest.mark.parametrize('field,value',[('microbatch',4),('gradient_accumulation',2),('drop_last',True)])
def test_changed_batch_is_rejected(recipe,field,value):
    from src.e2.config import validate_recipe
    recipe['batch'][field]=value
    with pytest.raises(ValueError):validate_recipe(recipe)

@pytest.mark.parametrize('field,value',[('experiment','E1-HR'),('kind','official_e1_run_config'),('representation_dimension',768),('view_execution','locals_first')])
def test_rehashed_other_stage_or_topology_is_rejected(run_config,field,value):
    from src.e2.config import validate_run,fingerprint
    run_config[field]=value
    run_config['config_sha256']=fingerprint({k:v for k,v in run_config.items() if k!='config_sha256'})
    with pytest.raises(ValueError):validate_run(run_config)

def test_other_stage_receipts_are_rejected(tmp_path):
    from src.e2.config import require_approval,CONTRACT_SHA256
    p=tmp_path/'receipt.json'
    p.write_text(json.dumps({'gate':'HR-G2','approved':True,'experiment':'E1-HR','contract_sha256':CONTRACT_SHA256}))
    with pytest.raises(ValueError):require_approval(p,gate='E2-G2',bindings={})

def test_final_requires_locked_selection(recipe,run_config):
    from src.e2.config import resolve_run
    with pytest.raises(ValueError):resolve_run(recipe,stage='final',lr=1e-4,seed=2001,paths=run_config.get('paths',{}))

def test_acceptance_request_never_passes_official_validation(recipe,run_config):
    from src.e2.config import validate_run,validate_acceptance_request,acceptance_request
    request=acceptance_request(recipe,paths=run_config['paths'],target_authorization='target.json')
    assert validate_acceptance_request(request)['stage']=='acceptance'
    assert request['seed']==1001 and request['lr']==2.5e-5
    with pytest.raises(ValueError):validate_run(request)
    request['official_run']=True
    with pytest.raises(ValueError):validate_acceptance_request(request)

def test_json_freeze_and_duplicate_key_guards(tmp_path,recipe):
    from src.e2.config import CONTRACT_PATH,load_recipe,read_json
    p=tmp_path/'candidate.json';p.write_bytes(CONTRACT_PATH.read_bytes())
    with pytest.raises(ValueError):load_recipe(p)
    for text in ['{"x":1,"x":2}','{"x":NaN}']:
        p.write_text(text)
        with pytest.raises(ValueError):read_json(p)
