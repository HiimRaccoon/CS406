"""Synthetic evidence only; candidate sizes/seeds/hashes here are not research decisions."""
import copy
import json

import pytest

from src.pilot.config import fingerprint
from src.pilot.p0_feasibility import validate_resolution_evidence
from test_config import complete_config


def fallback_case(tmp_path):
    cfg = complete_config(exp="e1hr")
    cfg["resolution_policy"]["ordered_candidates"].append([1024, 448])
    cfg["resolution_policy"]["batch_policy"]["candidate_batches"].append({"microbatch": 1, "accumulation": 2})
    cfg["global_wh"] = [1024, 448]
    cfg["batch"] = {"microbatch": 1, "accumulation": 2}
    context = {"version": 2, "feasibility_mode": "criteria", "budget": cfg["budget"], "protocol_id": cfg["protocol_id"],
               "ordered_candidates": cfg["resolution_policy"]["ordered_candidates"],
               "criteria_id": cfg["resolution_policy"]["criteria_id"],
               "feasibility": cfg["feasibility"], "seed": cfg["seed"],
               "runtime": cfg["runtime"], "initialization": cfg["initialization"],
               "batch_policy": cfg["resolution_policy"]["batch_policy"],
               "dataset_root": "synthetic-root", "fingerprints": {"manifest.csv": "synthetic-split"},
               "preprocessing_sha256": "synthetic-preprocessing",
               "pretrained_sha256": cfg["expected_weight_sha256"],
               "identity": {"gpu_name": "synthetic-device", "source_sha256": "synthetic-source"}}
    report = {"purpose": "pilot", "stage": "p0", "experiment": "e1hr", "status": "fail",
              "failure_kind": "oom", "global_wh": [1344, 576],
              "resolution_policy": copy.deepcopy(cfg["resolution_policy"]),
              "batch": {"microbatch": 2, "accumulation": 1}, "effective_batch": 2,
              "feasibility": copy.deepcopy(cfg["feasibility"]), "fallback_eligible": True,
              "feasibility_context": copy.deepcopy(context), "feasibility_fingerprint": fingerprint(context)}
    path = tmp_path / "prior.json"
    cfg["resolution_policy"]["prior_failure_reports"] = [str(path)]
    return cfg, context, report, path


@pytest.mark.parametrize("field", ["protocol_id", "ordered_candidates", "criteria_id", "feasibility", "seed",
                                  "runtime", "initialization", "batch_policy", "dataset_root", "fingerprints",
                                  "preprocessing_sha256", "pretrained_sha256", "identity"])
def test_reject_failure_from_different_reviewed_context(tmp_path, field):
    cfg, context, report, path = fallback_case(tmp_path)
    report["feasibility_context"][field] = "different"
    # Even a self-consistent hash from another protocol is not acceptable.
    report["feasibility_fingerprint"] = fingerprint(report["feasibility_context"])
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="feasibility"):
        validate_resolution_evidence(cfg, context)


@pytest.mark.parametrize("mutation", ["oversized_batch", "forged_fingerprint", "missing_context", "wrong_effective_batch"])
def test_reject_unbound_or_oversized_oom_evidence(tmp_path, mutation):
    cfg, context, report, path = fallback_case(tmp_path)
    if mutation == "oversized_batch":
        report["batch"] = {"microbatch": 64, "accumulation": 1}
    elif mutation == "forged_fingerprint":
        report["feasibility_fingerprint"] = "bad"
    elif mutation == "missing_context":
        del report["feasibility_context"]
    else:
        report["effective_batch"] = 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_resolution_evidence(cfg, context)


@pytest.mark.parametrize("failure_kind", ["oom", "criteria"])
def test_accept_reviewed_compatible_batch_policy_without_search(tmp_path, failure_kind):
    cfg, context, report, path = fallback_case(tmp_path)
    report["failure_kind"] = failure_kind
    path.write_text(json.dumps(report))
    validate_resolution_evidence(cfg, context)
