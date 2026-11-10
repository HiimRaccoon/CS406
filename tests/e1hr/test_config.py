import copy
import hashlib
import json

import pytest


def test_exact_hr_freeze_loads_and_is_allocation_free(recipe):
    from src.e1hr.config import CONTRACT_PATH, CONTRACT_SHA256, load_recipe
    assert hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() == CONTRACT_SHA256
    assert load_recipe() == recipe


@pytest.mark.parametrize('change', [
    lambda r: r['scope'].update(experiment='E1'),
    lambda r: r['preprocessing'].update(output_wh=[896,384]),
    lambda r: r['batch'].update(microbatch=4),
    lambda r: r['budget'].update(max_epochs=7),
    lambda r: r['budget'].update(early_stopping=True),
    lambda r: r['campaign'].update(lr_candidates=[1e-4]*4),
    lambda r: r['campaign']['final_seeds'].__setitem__(0,1001),
    lambda r: r['optimizer'].update(eps=float('nan')),
    lambda r: r['checkpoint'].update(tolerance=0),
    lambda r: r.pop('runtime'),
])
def test_rejects_any_changed_frozen_control(recipe, change):
    from src.e1hr.config import validate_recipe
    change(recipe)
    with pytest.raises(ValueError):
        validate_recipe(recipe)


def test_hr_contract_rejects_e1_and_rehashed_wrong_resolution(run_config):
    from src.e1hr.config import fingerprint, validate_run
    for key,value in [('experiment','E1'),('resolution_wh',[896,384]),('kind','official_e1_run_config')]:
        bad=copy.deepcopy(run_config); bad[key]=value
        bad['config_sha256']=fingerprint({k:v for k,v in bad.items() if k!='config_sha256'})
        with pytest.raises(ValueError): validate_run(bad)


def test_roles_grid_explicit_paths_and_immutable_budget(run_config,recipe):
    from src.e1hr.config import resolve_run,validate_run
    assert run_config['run_id']=='e1hr-tuning-lr0.00010000-seed1001'
    assert run_config['budget']==recipe['budget']
    assert validate_run(run_config)==run_config
    for stage,seed,lr in [('test',1001,1e-4),('tuning',2001,1e-4),('tuning',True,1e-4),('tuning',1001,.1)]:
        with pytest.raises(ValueError): resolve_run(recipe,stage=stage,seed=seed,lr=lr,paths=run_config['paths'])
    with pytest.raises(ValueError): resolve_run(recipe,stage='tuning',seed=1001,lr=1e-4,paths={})


def test_exact_file_sidecar_and_hr_receipt_are_required(recipe,tmp_path):
    from src.e1hr.config import CONTRACT_PATH,CONTRACT_SHA256,load_recipe,require_approval
    p=tmp_path/'e1hr_numeric_contract_v1.json'; p.write_bytes(CONTRACT_PATH.read_bytes())
    with pytest.raises(ValueError): load_recipe(p)
    p.with_suffix('.sha256').write_text(CONTRACT_SHA256+'  '+p.name+'\n')
    assert load_recipe(p)==recipe
    p.write_text(json.dumps(recipe))
    with pytest.raises(ValueError): load_recipe(p)
    q=tmp_path/'approval.json'; q.write_text(json.dumps({'gate':'G1','approved':True,'contract_sha256':CONTRACT_SHA256}))
    with pytest.raises(ValueError): require_approval(q,'HR-G1')


def test_duplicate_and_nonfinite_json_are_rejected(tmp_path):
    from src.e1hr.config import read_json
    p=tmp_path/'bad.json'
    for text in ['{"seed":1001,"seed":2001}','{"lr":NaN}']:
        p.write_text(text)
        with pytest.raises(ValueError): read_json(p)
