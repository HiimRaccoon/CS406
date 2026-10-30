import json

import pytest

from src.pilot import runtime
from src.preprocessing.geoguessr_v1 import PREPROCESS_SHA256
from test_config import complete_config


def test_p0_requires_target_hash_audit_and_no_test_access(tmp_path, manifest, monkeypatch):
    monkeypatch.setattr(runtime, "environment", lambda: {"packages": {"test": "only"}})
    cfg = complete_config()
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


def test_p1_requires_all_three_profiles_and_fair_batch_policy(tmp_path, manifest, monkeypatch):
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
        p.write_text(json.dumps({"stage": "p0", "status": "pass", "experiment": exp,
                                "full_epoch_measured": True, "fingerprints": manifest.fingerprints,
                                "preprocessing_sha256": PREPROCESS_SHA256, "effective_batch": 2,
                                "runtime_policy": cfg["runtime"], "seed": cfg["seed"],
                                "initialization_policy": cfg["initialization"],
                                "global_wh": [896, 384], "batch": cfg["batch"]}))
        paths.append(str(p))
    cfg["gates"]["p0_summaries"] = paths
    assert len(runtime.validate_gates(cfg, manifest)) == 3
    bad = json.loads((tmp_path / "e2.json").read_text())
    bad["effective_batch"] = 10  # view count must not inflate original-image batch
    (tmp_path / "e2.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="batch/runtime"):
        runtime.validate_gates(cfg, manifest)
