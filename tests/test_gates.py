import json
import copy

import pytest

from src.pilot import runtime
from src.pilot.config import fingerprint
from src.preprocessing.geoguessr_v1 import PREPROCESS_SHA256
from test_config import complete_config


@pytest.mark.parametrize("mode", ["criteria", "measurement_only"])
def test_p0_requires_target_hash_audit_and_no_test_access(tmp_path, manifest, monkeypatch, mode):
    monkeypatch.setattr(runtime, "environment", lambda: {"packages": {"test": "only"}})
    cfg = complete_config()
    cfg["feasibility"]["mode"] = mode
    cfg["paths"]["data_root"] = str(tmp_path.resolve())
    path = tmp_path / "conformance.json"
    cfg["gates"]["conformance_report"] = str(path)
    report = {"status": "pass", "preprocessing_sha256": PREPROCESS_SHA256,
              "fingerprints": manifest.fingerprints, "test_images_opened": 0,
              "dataset_root": str(tmp_path.resolve()), "environment": {"packages": {"test": "only"}},
              "image_hash_audit": None}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="verify-all"):
        runtime.validate_gates(cfg, manifest)
    report["image_hash_audit"] = {"train": 34997, "val": 7497}
    path.write_text(json.dumps(report))
    assert runtime.validate_gates(cfg, manifest) == []
    report["test_images_opened"] = 1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="conformance"):
        runtime.validate_gates(cfg, manifest)


@pytest.mark.parametrize("mutation", [None, "measurement", "status_only_promoted", "mode_only_promoted",
                                     "all_modes_promoted_null_constraints", "missing_mode", "missing_context_mode",
                                     "forged_fingerprint", "short_window", "forged_full_epoch"])
def test_p1_requires_all_three_profiles_and_fair_batch_policy(tmp_path, manifest, monkeypatch, mutation):
    monkeypatch.setattr(runtime, "environment", lambda: {"packages": {}})
    cfg = complete_config("p1")
    cfg["paths"]["data_root"] = str(tmp_path.resolve())
    path = tmp_path / "conformance.json"
    path.write_text(json.dumps({"status": "pass", "preprocessing_sha256": PREPROCESS_SHA256,
                               "fingerprints": manifest.fingerprints, "test_images_opened": 0,
                               "dataset_root": str(tmp_path.resolve()), "environment": {"packages": {}},
                               "image_hash_audit": {"train": 34997, "val": 7497}}))
    cfg["gates"]["conformance_report"] = str(path)
    cfg["gates"]["p0_summaries"] = []
    paths = []
    for exp in ("e1", "e1hr", "e2"):
        p = tmp_path / f"{exp}.json"
        feasibility = complete_config()["feasibility"]
        context = {"version": 2, "feasibility_mode": "criteria", "feasibility": copy.deepcopy(feasibility),
                   "budget": {"full_epoch": True}}
        p.write_text(json.dumps({"purpose": "pilot", "stage": "p0", "status": "pass", "experiment": exp,
                                "feasibility": feasibility, "feasibility_context": context,
                                "feasibility_fingerprint": fingerprint(context),
                                "full_epoch_measured": True, "fingerprints": manifest.fingerprints,
                                "preprocessing_sha256": PREPROCESS_SHA256, "effective_batch": 2,
                                "runtime_policy": cfg["runtime"], "seed": cfg["seed"],
                                "initialization_policy": cfg["initialization"],
                                "global_wh": [896, 384], "batch": cfg["batch"]}))
        paths.append(str(p))
    cfg["gates"]["p0_summaries"] = paths
    assert len(runtime.validate_gates(cfg, manifest)) == 3
    bad = json.loads((tmp_path / "e2.json").read_text())
    if mutation is not None:
        if mutation in {"measurement", "status_only_promoted", "mode_only_promoted", "all_modes_promoted_null_constraints"}:
            bad["status"] = "measured" if mutation == "measurement" else "pass"
            bad["feasibility"] = {"mode": "measurement_only", "max_peak_reserved_gib": None,
                                  "min_images_per_second": None, "max_epoch_seconds": None}
            bad["feasibility_context"]["feasibility_mode"] = "measurement_only"
            bad["feasibility_context"]["feasibility"] = copy.deepcopy(bad["feasibility"])
            if mutation in {"mode_only_promoted", "all_modes_promoted_null_constraints"}:
                bad["feasibility"]["mode"] = "criteria"
            if mutation == "all_modes_promoted_null_constraints":
                bad["feasibility_context"]["feasibility_mode"] = "criteria"
                bad["feasibility_context"]["feasibility"]["mode"] = "criteria"
            bad["feasibility_fingerprint"] = fingerprint(bad["feasibility_context"])
        elif mutation == "missing_mode":
            del bad["feasibility"]["mode"]
        elif mutation == "missing_context_mode":
            del bad["feasibility_context"]["feasibility_mode"]
            bad["feasibility_fingerprint"] = fingerprint(bad["feasibility_context"])
        elif mutation == "forged_fingerprint":
            bad["feasibility_fingerprint"] = "forged"
        else:
            bad["full_epoch_measured"] = mutation == "forged_full_epoch"
            bad["feasibility_context"]["budget"]["full_epoch"] = False
            bad["feasibility_fingerprint"] = fingerprint(bad["feasibility_context"])
        (tmp_path / "e2.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            runtime.validate_gates(cfg, manifest)
        return
    bad["effective_batch"] = 10  # view count must not inflate original-image batch
    (tmp_path / "e2.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="batch/runtime"):
        runtime.validate_gates(cfg, manifest)
