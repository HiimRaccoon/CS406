import copy
import json

import pytest


def test_frozen_recipe_and_recorded_g1_load_without_torch_allocation(recipe):
    from src.e1.config import CONTRACT_PATH, load_recipe, validate_recipe
    assert load_recipe(CONTRACT_PATH) == recipe
    assert validate_recipe(recipe) == recipe


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("runtime"), lambda r: r.update(purpose="pilot"),
    lambda r: r["campaign"].update(lr_candidates=[1e-4] * 4),
    lambda r: r["campaign"]["final_seeds"].__setitem__(0, 1001),
    lambda r: r["budget"].update(max_epochs=3), lambda r: r["budget"].update(early_stopping=True),
    lambda r: r["checkpoint"].update(tolerance=0, tie_break="nll"),
    lambda r: r["optimizer"]["groups"]["convnext_layer_scale_parameters"].update(weight_decay=.05),
    lambda r: r["model"]["pretrained"].update(sha256=None),
    lambda r: r["optimizer"].update(eps=float("nan")),
    lambda r: r["loss"].update(balanced_sampling=True),
    lambda r: r["deployment"].update(qos="wrong"),
])
def test_missing_overridden_or_pilot_values_are_rejected(recipe, mutation):
    from src.e1.config import validate_recipe
    mutation(recipe)
    with pytest.raises(ValueError):
        validate_recipe(recipe)


def test_modified_recipe_bytes_and_missing_sidecar_rejected(recipe, tmp_path):
    from src.e1.config import load_recipe
    p = tmp_path / "contract.json"
    p.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        load_recipe(p)


def test_resolution_enforces_roles_grid_and_explicit_paths(recipe, tmp_path):
    from src.e1.config import resolve_run, validate_run
    paths = {"data_root": str(tmp_path), "weights": str(tmp_path / "weights"), "runs_dir": str(tmp_path / "runs")}
    a = resolve_run(recipe, stage="tuning", lr=1e-4, seed=1001, paths=paths)
    b = resolve_run(recipe, stage="tuning", lr=1e-4, seed=1001, paths={**paths, "runs_dir": str(tmp_path / "other")})
    assert a["scientific_sha256"] == b["scientific_sha256"]
    assert a["config_sha256"] != b["config_sha256"]
    for stage, seed, lr in [("tuning", 2001, 1e-4), ("tuning", 3406, 1e-4), ("test", 1001, 1e-4), ("tuning", 1001, .1)]:
        with pytest.raises(ValueError):
            resolve_run(recipe, stage=stage, lr=lr, seed=seed, paths=paths)
    with pytest.raises(ValueError):
        resolve_run(recipe, stage="tuning", lr=1e-4, seed=1001, paths={})
    bad = copy.deepcopy(a)
    bad["seed"] = 2001
    with pytest.raises(ValueError):
        validate_run(bad)
