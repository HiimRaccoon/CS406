import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.pilot.config import ConfigError, ROOT, load_config, validate
from src.pilot.p0_feasibility import validate_resolution_evidence


def complete_config(stage="p0", exp="e1"):
    c = json.loads((ROOT / f"configs/pilot/{stage}_{exp}.json").read_text())
    # Synthetic test values only: never an endorsed training recipe.
    c.update(protocol_id="unit-test-only", seed=17, global_wh=[1344, 576] if exp == "e1hr" else [896, 384])
    c["expected_weight_sha256"] = hashlib.sha256(b"synthetic fixture only; not pretrained weights").hexdigest()
    c["paths"].update(data_root="unused", weights="unused", runs_dir="unused")
    c["batch"] = {"microbatch": 2, "accumulation": 1}
    c["initialization"].update(method="linear_default", seed=19)
    c["optimizer"].update(lr=.001, betas=[.9, .999], eps=1e-8)
    c["augmentation"]["enabled"] = False
    c["clipping"]["enabled"] = False
    c["runtime"].update(num_workers=0, cpu_threads=1, pin_memory=False, persistent_workers=False,
                         tf32_matmul=False, tf32_cudnn=False, matmul_precision="highest",
                         cudnn_benchmark=False, cudnn_deterministic=True, deterministic_algorithms=True,
                         deterministic_warn_only=False, torch_compile=False)
    c["runtime"]["amp"].update(enabled=False, dtype="float32")
    c["runtime"]["scaler"]["enabled"] = False
    c["gates"]["conformance_report"] = "unused"
    if stage == "p0":
        c["budget"] = {"warmup_updates": 1, "measure_updates": 1, "full_epoch": True}
        c["feasibility"] = {"mode": "criteria", "max_peak_reserved_gib": 30, "min_images_per_second": 1, "max_epoch_seconds": 3600}
    else:
        c["budget"]["max_epochs"] = 2
        c["scheduler"].update(warmup_updates=1, start_factor=.1, min_lr=1e-6)
        c["checkpoint"].update(tolerance=1e-6, tie_break="earlier")
        c["gates"]["p0_summaries"] = ["a", "b", "c"]
    if exp == "e1hr":
        c["resolution_policy"]["criteria_id"] = "unit-test-only"
        c["resolution_policy"]["batch_policy"] = {"effective_batch": 2,
                                                  "candidate_batches": [{"microbatch": 2, "accumulation": 1}]}
    return c


@pytest.mark.parametrize("stage", ["p0", "p1"])
@pytest.mark.parametrize("exp", ["e1", "e1hr", "e2"])
def test_templates_fail_until_reviewed(stage, exp, monkeypatch):
    for key in ("CS406_DATA_ROOT", "CS406_WEIGHTS_PATH", "CS406_RUNS_DIR"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigError, match="Pending values"):
        load_config(ROOT / f"configs/pilot/{stage}_{exp}.json")
    validate(complete_config(stage, exp))


def test_search_and_weighting_forbidden():
    for field in ("class_weights", "lr_grid", "test_split"):
        cfg = complete_config()
        cfg[field] = []
        with pytest.raises(ConfigError, match="outside"):
            validate(cfg)


def test_hr_no_automatic_fallback_or_skipped_candidate(tmp_path):
    cfg = complete_config(exp="e1hr")
    validate_resolution_evidence(cfg)
    # Artificial sizes only test validation logic, not real fallback proposals.
    cfg["resolution_policy"]["ordered_candidates"].append([1024, 448])
    cfg["resolution_policy"]["batch_policy"]["candidate_batches"].append({"microbatch": 1, "accumulation": 2})
    cfg["global_wh"] = [1024, 448]
    cfg["batch"] = {"microbatch": 1, "accumulation": 2}
    with pytest.raises(ConfigError, match="failure reports"):
        validate(cfg)
    p = tmp_path / "failure.json"
    p.write_text(json.dumps({"stage": "p0", "experiment": "e1hr", "status": "fail",
                            "failure_kind": "infrastructure_or_numerical", "global_wh": [1344, 576],
                            "resolution_policy": cfg["resolution_policy"]}))
    cfg["resolution_policy"]["prior_failure_reports"] = [str(p)]
    with pytest.raises(ValueError, match="genuine"):
        validate_resolution_evidence(cfg)


@pytest.mark.parametrize("stage", ["p0", "p1"])
@pytest.mark.parametrize("digest", [None, "", "983f1562", "g" * 64, "a" * 63, 123])
def test_full_pretrained_sha_required_before_model_build(stage, digest):
    cfg = complete_config(stage)
    cfg["expected_weight_sha256"] = digest
    with pytest.raises(ConfigError, match="expected_weight_sha256"):
        validate(cfg)


def test_normalize_full_sha_and_conformance_does_not_require_weights():
    cfg = complete_config()
    digest = cfg["expected_weight_sha256"]
    cfg["expected_weight_sha256"] = " " + digest.upper() + "\n"
    assert validate(cfg)["expected_weight_sha256"] == digest
    c = json.loads((ROOT / "configs/pilot/conformance.json").read_text())
    c["paths"]["data_root"] = "unused"
    validate(c)


def test_hr_batch_must_match_reviewed_policy():
    cfg = complete_config(exp="e1hr")
    cfg["batch"] = {"microbatch": 64, "accumulation": 1}
    with pytest.raises(ConfigError, match="batch"):
        validate(cfg)
