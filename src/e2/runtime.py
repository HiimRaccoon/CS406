"""Consumed-source identity and current UIT inputs, separate from audit tooling."""
from __future__ import annotations

import ast
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .config import CONTRACT_SHA256, ROOT, fingerprint, load_recipe, read_json, sha256_file, validate_run, validate_acceptance_request, topology
from .data import E2Dataset, FixedManifest, make_loaders
from .engine import UpdateScheduler
from .model import build_e2, build_optimizer


def git_audit() -> dict:
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError('Actual clean Git checkout required') from exc
    if dirty:
        raise ValueError('Actual clean Git checkout required')
    return {'clean_git_commit': commit, 'git_clean': True}


def _review_imports(paths: list[str]) -> None:
    """Reject newly consumed repository imports outside the reviewed explicit map."""
    allowed = set(paths)
    for rel in paths:
        if not rel.endswith('.py'):
            continue
        package = list(Path(rel).with_suffix('').parts[:-1])
        for node in ast.walk(ast.parse((ROOT / rel).read_text(encoding='utf-8'))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                parts = prefix + (node.module.split('.') if node.module else [])
                modules = ['.'.join(parts)]
                if node.module is None:
                    modules += ['.'.join(parts + [a.name]) for a in node.names]
            for module in modules:
                if not module.startswith('src.') and module != 'src':
                    continue
                parts = module.split('.')
                for count in range(1,len(parts)+1):
                    stem = Path(*parts[:count])
                    candidates = [stem.with_suffix('.py')] if count == len(parts) else []
                    candidates.append(stem / '__init__.py')
                    for p in candidates:
                        if (ROOT / p).is_file() and p.as_posix() not in allowed:
                            raise ValueError(f'Consumed import outside reviewed closure: {p.as_posix()}')


def source_identity(recipe: dict | None = None) -> dict:
    recipe = load_recipe() if recipe is None else recipe
    paths = recipe['identity_and_evidence']['source_fingerprint_closure']
    if len(paths) != len(set(paths)) or any('*' in p or p.startswith('tests/') for p in paths):
        raise ValueError('Explicit consumed-source closure required')
    hashes = {p: sha256_file(ROOT / p) for p in sorted(set(paths))}
    _review_imports(paths)
    return {'scientific_runtime': {'consumed_source_files': hashes, 'consumed_source_sha256': fingerprint(hashes)},
            'audit': git_audit()}


def acceptance_inventory(recipe: dict | None = None) -> dict:
    """Audit only; never folded into the scientific/runtime equality projection."""
    recipe = load_recipe() if recipe is None else recipe
    spec = recipe['identity_and_evidence']['acceptance_test_evidence_inventory']
    paths = list(spec['harness_files'])
    paths += ['tests/e2/test_runtime.py', 'tests/e2/helpers.py']
    paths += [p for p in spec['optional_package_markers_if_present'] if (ROOT / p).is_file()]
    return {p: {'bytes': (ROOT/p).stat().st_size, 'sha256': sha256_file(ROOT/p), 'role': 'acceptance_test_tooling'}
            for p in sorted(set(paths))}


def _software() -> dict:
    return {'python':platform.python_version(),
            **{name:importlib.metadata.version(name) for name in ('torch','torchvision','Pillow','numpy','pandas','scikit-learn')},
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}


def _allocation(recipe: dict) -> dict:
    job = os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():
        raise ValueError('HR GPU work requires a reviewed Slurm allocation')
    text = subprocess.check_output(['scontrol','show','job','--oneliner',job],text=True)
    fields = dict(word.split('=',1) for word in text.split() if '=' in word)
    d = recipe['deployment']
    if (fields.get('Account') != d['account'] or fields.get('QOS') != d['qos']
            or fields.get('Partition') != d['partition'] or fields.get('TimeLimit') != d['job_walltime']
            or fields.get('NumTasks') != str(d['ntasks'])
            or os.environ.get('SLURM_CPUS_PER_TASK') != str(d['cpus_per_task'])
            or os.environ.get('SLURM_MEM_PER_NODE') != str(d['slurm_mem_mib'])
            or 'gres/gpu:2g.35gb=1' not in fields.get('ReqTRES','').split(',')):
        raise ValueError('Actual allocation differs from the frozen E2 policy')
    return {'job_id':job,'record':fields}


def effective_runtime() -> dict:
    return {'cpu_threads':torch.get_num_threads(),'matmul_precision':torch.get_float32_matmul_precision(),
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}


def configure_runtime(seed: int) -> dict:
    random.seed(seed); np.random.seed(seed % 2**32)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True,warn_only=False)
    actual = effective_runtime()
    expected = {'cpu_threads':2,'matmul_precision':'highest','tf32_matmul':False,'tf32_cudnn':False,
                'cudnn_benchmark':False,'cudnn_deterministic':True,'deterministic_algorithms':True,'warn_only':False}
    if actual != expected:
        raise ValueError('Effective E2 Torch runtime mismatch')
    return actual


def audit_current_images(data_root: Path) -> dict:
    return FixedManifest().verify_images(data_root)


def _e2_conformance(manifest, config) -> dict:
    dataset = E2Dataset(manifest,config['paths']['data_root'],'train',config['seed'],{'train_enabled':False})
    members = []
    expected = {'global':[3,384,896],'local':[4,3,320,320]}
    for layout in ((1536,662),(1030,703)):
        index = next(i for i,r in enumerate(dataset.records) if (r.width,r.height) == layout)
        value,_,path = dataset[index]
        again = dataset[index][0]
        if any(list(value[k].shape) != shape or value[k].dtype != torch.float32
               or not torch.isfinite(value[k]).all() or not torch.equal(value[k],again[k])
               for k,shape in expected.items()):
            raise ValueError('Current frozen E2 preprocessing conformance failed')
        members.append({'relative_path':path,'sha256':dataset.records[index].sha256,'raw_wh':list(layout)})
    return {'view_shapes':expected,'train_members':members,'test_images_opened':0}


def device_identity() -> dict:
    if not torch.cuda.is_available():
        raise ValueError('Accepted UIT CUDA device required')
    props = torch.cuda.get_device_properties(0)
    driver = subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip().splitlines()[0]
    return {'name':props.name,'total_bytes':props.total_memory,'driver':driver}


def inspect_target_inputs(config: dict) -> dict:
    """No model allocation; callers enforce acceptance or official stage permission."""
    config = (validate_acceptance_request(config) if config.get('kind') == 'e2_target_acceptance_request_v1' else validate_run(config))
    recipe,paths = config['recipe'],config['paths']
    d = recipe['deployment']
    if os.path.abspath(sys.executable) != d['python_executable']:
        raise ValueError('Reviewed Python executable required before target access')
    for key,pinned in [('data_root','dataset'),('weights','pretrained')]:
        if paths[key] != d['paths'][pinned]:
            raise ValueError('E2 paths differ from reviewed target')
    roots = [Path(paths[k]).resolve() for k in ('runs_dir','evidence_root')]
    protected = [ROOT/'runs'/s for s in ('E1','E1-HR','pilot')]
    protected += [ROOT/'reports'/s for s in ('E1','E1-HR','pilot')]
    protected += [Path(paths['data_root']).resolve()]
    if (roots[0] == roots[1] or any(a.is_relative_to(b) or b.is_relative_to(a)
            for a in roots for b in protected)
            or any(a == ROOT or ROOT.is_relative_to(a) for a in roots)):
        raise ValueError('Explicit separate E2 output roots required')
    required_env = {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':str(config['seed'])}
    if any(os.environ.get(k) != v for k,v in required_env.items()) or os.environ.get('NVIDIA_TF32_OVERRIDE') not in {None,'0'}:
        raise ValueError('Export exact E2 runtime variables before Python startup')
    software = _software()
    if software != recipe['runtime']['software_lock']:
        raise ValueError('Actual software differs from frozen E2 policy')
    allocation = _allocation(recipe)
    source = source_identity(recipe)
    manifest = FixedManifest()
    pre = recipe['preprocessing']
    if sha256_file(ROOT/pre['implementation_path']) != pre['implementation_sha256']:
        raise ValueError('Frozen preprocessing module changed')
    weight = Path(paths['weights'])
    if weight.name != recipe['model']['pretrained']['filename'] or sha256_file(weight) != recipe['model']['pretrained']['sha256']:
        raise ValueError('Real pinned offline weights required')
    report_path = Path(paths['conformance_report'])
    report = read_json(report_path)
    packages = {k:software[k] for k in ('torch','torchvision','Pillow','numpy','pandas','scikit-learn')}
    if (report.get('status') != 'pass' or report.get('test_images_opened') != 0
            or report.get('dataset_root') != paths['data_root'] or report.get('fingerprints') != manifest.fingerprints
            or report.get('preprocessing_sha256') != pre['config_sha256']
            or report.get('image_hash_audit') != {'train':34997,'val':7497}
            or report.get('environment',{}).get('packages') != packages):
        raise ValueError('Passing frozen conformance for actual root/packages required')
    audit = manifest.verify_images(Path(paths['data_root']))
    current_e2 = _e2_conformance(manifest,config)
    actual_runtime = configure_runtime(config['seed'])
    device = device_identity()
    if device['name'] != d['expected_gpu_class']:
        raise ValueError('Wrong E2 target GPU class')
    return {'scientific_runtime':{
        **source['scientific_runtime'],**topology(recipe),
        'contract_sha256':CONTRACT_SHA256,'software':software,'effective_runtime':actual_runtime,'runtime_policy':recipe['runtime'],
        'device_and_driver':device,'reviewed_allocation_policy':{k:d[k] for k in ('partition','account','qos','gres','ntasks','cpus_per_task','slurm_mem_mib','job_walltime')},
        'dataset_root':paths['data_root'],'fingerprints':manifest.fingerprints,
        'preprocessing_config_sha256':pre['config_sha256'],'preprocessing_implementation_sha256':pre['implementation_sha256'],
        'pretrained_sha256':recipe['model']['pretrained']['sha256'],'accepted_conformance_sha256':sha256_file(report_path),
        'e2_preprocessing_conformance':current_e2,'evidence_kind':'official_uit_slurm'},
        'audit':{**source['audit'],'acceptance_test_evidence_inventory':acceptance_inventory(recipe),
                 'platform':platform.platform(),'python_executable':sys.executable,
                 'runtime_environment':{k:os.environ.get(k) for k in (*required_env,'NVIDIA_TF32_OVERRIDE')},
                 'allocation':allocation,'current_image_hash_audit':audit,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}}


def prepare(config: dict, checked: dict) -> dict:
    """Called only by authorized acceptance or the official runner after preflight."""
    recipe = config['recipe']; science = checked['scientific_runtime']
    if device_identity() != science['device_and_driver']:
        raise ValueError('Device/driver changed; renewed target acceptance required')
    actual = configure_runtime(config['seed'])
    if actual != science['effective_runtime']:
        raise ValueError('Effective runtime changed after target input inspection')
    model,metadata = build_e2(Path(config['paths']['weights']),science['pretrained_sha256'],{**recipe['initialization'],'seed':config['seed']})
    model.to('cuda')
    optimizer,groups = build_optimizer(model,{**recipe['optimizer'],'lr':config['lr']})
    manifest = FixedManifest()
    loaders = make_loaders(manifest,Path(config['paths']['data_root']),config)
    if len(loaders['train']) != recipe['batch']['updates_per_epoch']:
        raise ValueError('Natural E2 loader/update accounting mismatch')
    scheduler = UpdateScheduler(optimizer,recipe['scheduler'],recipe['budget']['planned_max_updates'])
    return dict(config=config,model=model,optimizer=optimizer,scheduler=scheduler,scaler=None,loaders=loaders,
                device=torch.device('cuda'),n_train=manifest.n_train_by_label,labels=manifest.labels,
                train_samples=manifest.n_train,val_samples=len(manifest.records('val')),identity=checked,
                environment={'software':science['software'],'device':science['device_and_driver'],
                             'allocation':checked['audit']['allocation'],'actual_torch_settings':actual,
                             'runtime_policy':recipe['runtime'],'audit':checked['audit']},
                model_metadata={**metadata,'optimizer_groups':groups})
