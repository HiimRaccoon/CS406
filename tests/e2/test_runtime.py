import copy
import hashlib
import json
from pathlib import Path

import pytest


def test_source_fingerprint_ignores_clean_docs_tests_commit_but_tracks_consumed_files(recipe,tmp_path,monkeypatch):
    import src.e2.runtime as runtime
    for rel in recipe['identity_and_evidence']['source_fingerprint_closure']:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    commit={'value':'a'*40}
    monkeypatch.setattr(runtime,'git_audit',lambda:{'clean_git_commit':commit['value'],'git_clean':True})
    first=runtime.source_identity(recipe)
    (tmp_path/'docs').mkdir();(tmp_path/'docs/note.md').write_text('new docs')
    (tmp_path/'tests').mkdir();(tmp_path/'tests/probe.py').write_text('new tests')
    commit['value']='b'*40
    second=runtime.source_identity(recipe)
    assert first['scientific_runtime']==second['scientific_runtime']
    assert first['audit']!=second['audit']
    (tmp_path/'src/e2/train.py').write_text('# changed runtime\n')
    assert runtime.source_identity(recipe)['scientific_runtime']!=first['scientific_runtime']


def test_new_unreviewed_transitive_import_is_rejected(recipe,tmp_path,monkeypatch):
    import src.e2.runtime as runtime
    for rel in recipe['identity_and_evidence']['source_fingerprint_closure']:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
    (tmp_path/'src/e2/train.py').write_text('from .unreviewed import train\n')
    (tmp_path/'src/e2/unreviewed.py').write_text('# helper\n')
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    monkeypatch.setattr(runtime,'git_audit',lambda:{'clean_git_commit':'a'*40,'git_clean':True})
    with pytest.raises(ValueError,match='closure'): runtime.source_identity(recipe)


def test_inspection_rejects_wrong_python_before_target_access(run_config,monkeypatch):
    import src.e2.runtime as runtime
    monkeypatch.setattr(runtime,'_allocation',lambda *a:pytest.fail('target access before interpreter verification'))
    with pytest.raises(ValueError,match='Python executable'):runtime.inspect_target_inputs(run_config)


def test_official_preflight_compares_projection_without_commit_equality(run_config,tmp_path,monkeypatch):
    import src.e2.train as train
    from src.e2.config import CONTRACT_SHA256,sha256_file
    from src.e2.checkpoint import atomic_json
    identity={'scientific_runtime':{'consumed_source_sha256':'a'*64,'runtime':'fp32'},'audit':{'clean_git_commit':'1'*40}}
    acceptance=Path(run_config['paths']['acceptance_record'])
    atomic_json(acceptance,{'gate':'E2-G2','approved':True,'experiment':'E2','views':run_config['views'],
               'contract_sha256':CONTRACT_SHA256,'identity':identity,'acceptance_evidence_sha256':'e'*64})
    evidence=tmp_path/'evidence.json';atomic_json(evidence,{'status':'pass','kind':'e2_target_acceptance_v1','experiment':'E2',
        'views':run_config['views'],'contract_sha256':CONTRACT_SHA256,'official_run':False,'test_images_opened':0,'identity':identity})
    receipt=json.loads(acceptance.read_text());receipt['acceptance_evidence']=str(evidence);receipt['acceptance_evidence_sha256']=sha256_file(evidence);atomic_json(acceptance,receipt)
    atomic_json(Path(run_config['paths']['authorization_record']),{'gate':'E2-G3a','approved':True,'experiment':'E2','views':run_config['views'],
               'contract_sha256':CONTRACT_SHA256,'acceptance_sha256':sha256_file(acceptance),
               'run_ids':[f'e2-tuning-lr{lr:.8f}-seed1001' for lr in run_config['recipe']['campaign']['lr_candidates']],
               'declarations':{f'e2-tuning-lr{lr:.8f}-seed1001':(run_config['config_sha256'] if lr==run_config['lr'] else 'd'*64) for lr in run_config['recipe']['campaign']['lr_candidates']}})
    current=copy.deepcopy(identity);current['audit']['clean_git_commit']='2'*40
    monkeypatch.setattr(train,'inspect_target_inputs',lambda c:copy.deepcopy(current))
    assert train.preflight(run_config)['audit']['clean_git_commit']=='2'*40
    current['scientific_runtime']['consumed_source_sha256']='b'*64
    with pytest.raises(ValueError): train.preflight(run_config)


def test_preflight_rejects_rehashed_e1_acceptance_evidence(run_config,tmp_path,monkeypatch):
    import src.e2.train as train
    from src.e2.config import CONTRACT_SHA256,sha256_file
    from src.e2.checkpoint import atomic_json
    identity={'scientific_runtime':{'source':'fixture'},'audit':{'clean_git_commit':'a'*40}}
    evidence=tmp_path/'evidence.json';atomic_json(evidence,{'status':'pass','kind':'e1_t10_gpu_smoke','identity':identity})
    p=Path(run_config['paths']['acceptance_record'])
    atomic_json(p,{'gate':'E2-G2','approved':True,'experiment':'E2','views':run_config['views'],'contract_sha256':CONTRACT_SHA256,
                  'identity':identity,'acceptance_evidence':str(evidence),'acceptance_evidence_sha256':sha256_file(evidence)})
    atomic_json(Path(run_config['paths']['authorization_record']),{'gate':'E2-G3a','approved':True,'experiment':'E2','views':run_config['views'],
               'contract_sha256':CONTRACT_SHA256,'acceptance_sha256':sha256_file(p),
               'run_ids':[f'e2-tuning-lr{lr:.8f}-seed1001' for lr in run_config['recipe']['campaign']['lr_candidates']],
               'declarations':{f'e2-tuning-lr{lr:.8f}-seed1001':(run_config['config_sha256'] if lr==run_config['lr'] else 'd'*64) for lr in run_config['recipe']['campaign']['lr_candidates']}})
    monkeypatch.setattr(train,'inspect_target_inputs',lambda c:identity)
    with pytest.raises(ValueError): train.preflight(run_config)


@pytest.mark.parametrize('change',['changed','missing','escaped'])
def test_current_data_audit_precedes_model_allocation(recipe,tmp_path,monkeypatch,change):
    import src.e2.runtime as runtime
    from src.e2.data import FixedManifest,Record
    data=tmp_path/'data';p=data/'Aland/a.jpg';p.parent.mkdir(parents=True);p.write_bytes(b'old')
    record=Record('../escape.jpg' if change=='escaped' else 'Aland/a.jpg','Aland',0,'train',hashlib.sha256(b'old').hexdigest(),1536,662)
    if change=='changed':p.write_bytes(b'new')
    if change=='missing':p.unlink()
    class Mini(FixedManifest):
        def __init__(self):self._records={'train':(record,),'val':()}
    monkeypatch.setattr(runtime,'FixedManifest',Mini)
    monkeypatch.setattr(runtime,'build_e2',lambda *a,**k:pytest.fail('model before audit'))
    with pytest.raises((ValueError,FileNotFoundError)): runtime.audit_current_images(data)


def test_acceptance_inventory_contains_actual_root_conftest_and_helpers(recipe):
    from src.e2.runtime import acceptance_inventory,ROOT
    from src.e2.config import sha256_file
    inventory=acceptance_inventory(recipe)
    for rel in ('tests/conftest.py','tests/e2/helpers.py','tests/e2/test_gpu_smoke.py'):
        assert inventory[rel]['sha256']==sha256_file(ROOT/rel)
        assert inventory[rel]['bytes']==(ROOT/rel).stat().st_size
        assert rel not in recipe['identity_and_evidence']['source_fingerprint_closure']


def test_actual_consumed_imports_fit_explicit_closure(recipe,monkeypatch):
    import src.e2.runtime as runtime
    monkeypatch.setattr(runtime,'git_audit',lambda:{'readiness_only':True,'not_target_acceptance':True})
    source=runtime.source_identity(recipe)
    assert len(source['scientific_runtime']['consumed_source_files'])==21
    assert 'src/e2/report.py' not in source['scientific_runtime']['consumed_source_files']


@pytest.mark.parametrize('field,value',[('Account','other'),('QOS','other'),('Partition','other'),
    ('TimeLimit','10:00:00'),('NumTasks','2'),('ReqTRES','gres/gpu:2g.35gb=2')])
def test_allocation_policy_is_checked_without_gpu(recipe,monkeypatch,field,value):
    import src.e2.runtime as runtime
    fields={'Account':'uit','QOS':'mig2-35','Partition':'normal','TimeLimit':'09:00:00',
            'NumTasks':'1','ReqTRES':'gres/gpu:2g.35gb=1'}
    monkeypatch.setenv('SLURM_JOB_ID','12345');monkeypatch.setenv('SLURM_CPUS_PER_TASK','4')
    monkeypatch.setenv('SLURM_MEM_PER_NODE','16384')
    monkeypatch.setattr(runtime.subprocess,'check_output',lambda *a,**k:' '.join(k+'='+v for k,v in fields.items()))
    assert runtime._allocation(recipe)['job_id']=='12345'
    fields[field]=value
    with pytest.raises(ValueError):runtime._allocation(recipe)


@pytest.mark.parametrize('field,value',[('status','failed'),('test_images_opened',1),
    ('dataset_root','changed'),('image_hash_audit',{'train':34997,'val':1}),('environment',{})])
def test_stale_conformance_rejected_before_model_allocation(recipe,run_config,tmp_path,monkeypatch,field,value):
    import src.e2.runtime as runtime
    from src.e2.config import resolve_run
    from src.e2.checkpoint import atomic_json
    d=recipe['deployment'];pre=recipe['preprocessing'];software=recipe['runtime']['software_lock']
    config=resolve_run(recipe,stage='tuning',lr=1e-4,seed=1001,paths={**run_config['paths'],
        'data_root':d['paths']['dataset'],'weights':d['paths']['pretrained']})
    monkeypatch.setattr(runtime.sys,'executable',d['python_executable'])
    # On Windows abspath adds the drive to the POSIX reviewed interpreter.
    monkeypatch.setattr(runtime.os.path,'abspath',lambda p:p)
    for k,v in {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'1001'}.items():monkeypatch.setenv(k,v)
    monkeypatch.delenv('NVIDIA_TF32_OVERRIDE',raising=False)
    monkeypatch.setattr(runtime,'_software',lambda:software)
    monkeypatch.setattr(runtime,'_allocation',lambda r:{'fixture':True})
    monkeypatch.setattr(runtime,'source_identity',lambda r:{'scientific_runtime':{},'audit':{}})
    class MetadataOnly:
        fingerprints={k:v['sha256'] for k,v in recipe['data']['artifacts'].items()}
        def verify_images(self,*a):pytest.fail('audit must follow conformance validation')
    monkeypatch.setattr(runtime,'FixedManifest',MetadataOnly)
    monkeypatch.setattr(runtime,'sha256_file',lambda p:pre['implementation_sha256'] if Path(p).suffix=='.py' else recipe['model']['pretrained']['sha256'])
    monkeypatch.setattr(runtime,'build_e2',lambda *a,**k:pytest.fail('model allocated'))
    monkeypatch.setattr(runtime,'device_identity',lambda:pytest.fail('device accessed'))
    report={'status':'pass','test_images_opened':0,'dataset_root':config['paths']['data_root'],
            'fingerprints':MetadataOnly.fingerprints,'preprocessing_sha256':pre['config_sha256'],
            'image_hash_audit':{'train':34997,'val':7497},'environment':{'packages':{k:software[k] for k in ('torch','torchvision','Pillow','numpy','pandas','scikit-learn')}}}
    report[field]=value;atomic_json(Path(config['paths']['conformance_report']),report)
    with pytest.raises(ValueError,match='conformance'):runtime.inspect_target_inputs(config)
