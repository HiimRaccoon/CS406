from pathlib import Path
import json
import pytest

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def recipe():
    return json.loads((ROOT/'docs/E2/e2_numeric_contract_v1.json').read_text(encoding='utf-8'))

@pytest.fixture
def run_config(recipe,tmp_path):
    from src.e2.config import resolve_run
    return resolve_run(recipe,stage='tuning',lr=1e-4,seed=1001,paths={
        'data_root':str(tmp_path/'data'),'weights':str(tmp_path/'convnext_tiny-983f1562.pth'),
        'runs_dir':str(tmp_path/'runs'),'evidence_root':str(tmp_path/'evidence'),
        'conformance_report':str(tmp_path/'conformance.json'),'acceptance_record':str(tmp_path/'g2.json'),
        'authorization_record':str(tmp_path/'g3a.json')})
