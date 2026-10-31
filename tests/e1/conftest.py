import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recipe():
    return json.loads((ROOT / "docs/E1/e1_numeric_contract_v1.json").read_text(encoding="utf-8"))


@pytest.fixture
def synthetic_recipe(recipe):
    # A disposable copy for rejection tests; never exported as a recipe.
    return copy.deepcopy(recipe)


@pytest.fixture
def run_config(recipe, tmp_path):
    from src.e1.config import resolve_run
    return resolve_run(recipe, stage="tuning", lr=1e-4, seed=1001,
                       paths={"data_root": str(tmp_path / "data"), "weights": str(tmp_path / "convnext_tiny-983f1562.pth"),
                              "runs_dir": str(tmp_path / "runs")})
