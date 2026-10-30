"""Mode/gate regression checks with synthetic CPU fixtures, never GPU measurements."""
import copy
import hashlib
import json
import sys

import pytest
import torch

from src.pilot import p0_feasibility as p0
from src.pilot.config import ConfigError, fingerprint, validate
from test_config import complete_config
from test_hr_fallback import fallback_case
from test_pilot_runners import setup_runner


def measurement_config(exp="e1"):
    cfg = complete_config(exp=exp)
    cfg["feasibility"] = {"mode": "measurement_only", "max_peak_reserved_gib": None,
                          "min_images_per_second": None, "max_epoch_seconds": None}
    return cfg


@pytest.mark.parametrize("mode", [None, "absent", "unknown"])
def test_p0_requires_explicit_valid_mode(mode):
    cfg = complete_config()
    if mode == "absent":
        del cfg["feasibility"]["mode"]
    else:
        cfg["feasibility"]["mode"] = mode
    with pytest.raises(ConfigError, match="mode"):
        validate(cfg)


@pytest.mark.parametrize("field", ["max_peak_reserved_gib", "min_images_per_second", "max_epoch_seconds"])
def test_criteria_still_requires_positive_constraints(field):
    cfg = complete_config()
    cfg["feasibility"][field] = None
    with pytest.raises(ConfigError, match=field):
        validate(cfg)
    cfg["feasibility"][field] = 0
    with pytest.raises(ConfigError):
        validate(cfg)


@pytest.mark.parametrize("exp", ["e1", "e1hr", "e2"])
def test_measurement_config_accepts_null_thresholds_only(exp):
    validate(measurement_config(exp))


@pytest.mark.parametrize("field", ["expected_weight_sha256", "seed", "batch.microbatch", "runtime.device",
                                  "gates.conformance_report", "initialization.method", "budget.measure_updates"])
def test_measurement_does_not_bypass_other_required_fields(field):
    cfg = measurement_config()
    parent = cfg
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = None
    with pytest.raises(ConfigError, match=field):
        validate(cfg)


def test_measurement_preserves_no_test_no_search_and_reviewed_hr_batch():
    for key in ("test_split", "lr_grid", "oversampling"):
        cfg = measurement_config()
        cfg[key] = []
        with pytest.raises(ConfigError, match="outside"):
            validate(cfg)
    cfg = measurement_config("e1hr")
    cfg["batch"]["microbatch"] = 64
    with pytest.raises(ConfigError, match="batch"):
        validate(cfg)


def prepare_cpu_run(monkeypatch, tmp_path, manifest):
    setup_runner(monkeypatch, p0, manifest)
    cfg = measurement_config()
    cfg["paths"]["runs_dir"] = str(tmp_path)
    weights = tmp_path / "convnext_tiny-983f1562.pth"
    weights.write_bytes(b"synthetic CPU mode fixture; not pretrained")
    cfg["paths"]["weights"] = str(weights)
    cfg["expected_weight_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setattr(p0, "load_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(sys, "argv", ["pilot", "--config", "unused", "--run-id", "mode-test"])
    return cfg


def test_measurement_success_logs_without_acceptance_and_exits_zero(monkeypatch, tmp_path, manifest):
    cfg = prepare_cpu_run(monkeypatch, tmp_path, manifest)
    builds = []
    original = p0.build_model
    def one_build(*args):
        builds.append(args[0])
        return original(*args)
    monkeypatch.setattr(p0, "build_model", one_build)
    p0.main()
    result = json.loads((tmp_path / "mode-test/run_summary.json").read_text())
    assert result["status"] == "measured" and result["assessment"] == "not_assessed"
    assert result["fallback_eligible"] is False and result["failures"] == []
    assert result["timing"]["steady_state"]["samples"] == 2
    assert result["timing"]["full_epoch"]["samples"] == 7
    assert set(result["memory"]) == {"measured_peak_allocated_bytes", "measured_peak_reserved_bytes",
                                     "epoch_peak_allocated_bytes", "epoch_peak_reserved_bytes"}
    assert result["effective_batch"] == 2 and result["test_images_opened"] == 0
    assert result["feasibility"]["mode"] == "measurement_only"
    assert result["feasibility_context"]["feasibility_mode"] == "measurement_only"
    assert result["feasibility_fingerprint"] == fingerprint(result["feasibility_context"])
    assert builds == ["e1"]  # one declared config, never a batch/architecture retry loop


@pytest.mark.parametrize("error", [torch.cuda.OutOfMemoryError, RuntimeError])
def test_measurement_failure_is_never_resolution_evidence(monkeypatch, tmp_path, manifest, error):
    prepare_cpu_run(monkeypatch, tmp_path, manifest)
    def fail_allocation(*args):
        raise error("synthetic failure; no GPU")
    monkeypatch.setattr(p0, "build_model", fail_allocation)
    with pytest.raises(error):
        p0.main()
    result = json.loads((tmp_path / "mode-test/run_summary.json").read_text())
    assert result["status"] == "fail" and result["fallback_eligible"] is False
    assert result["failure_kind"] == ("oom" if error is torch.cuda.OutOfMemoryError else "infrastructure_or_numerical")
    assert result["feasibility"]["mode"] == "measurement_only"
    assert result["feasibility_context"]["feasibility_mode"] == "measurement_only"


@pytest.mark.parametrize("mutation", ["measured_status", "measurement_mode", "ineligible", "missing_mode",
                                     "missing_context_mode", "forged_mode_with_stale_fingerprint"])
def test_hr_rejects_noncriteria_or_unbound_mode(tmp_path, mutation):
    cfg, context, report, path = fallback_case(tmp_path)
    if mutation == "measured_status":
        report["status"] = "measured"
    elif mutation == "measurement_mode":
        report["feasibility"]["mode"] = "measurement_only"
        report["feasibility_context"]["feasibility"]["mode"] = "measurement_only"
        report["feasibility_context"]["feasibility_mode"] = "measurement_only"
        report["feasibility_fingerprint"] = fingerprint(report["feasibility_context"])
    elif mutation == "ineligible":
        report["fallback_eligible"] = False
    elif mutation == "missing_mode":
        del report["feasibility"]["mode"]
    elif mutation == "missing_context_mode":
        del report["feasibility_context"]["feasibility_mode"]
        report["feasibility_fingerprint"] = fingerprint(report["feasibility_context"])
    else:
        report["feasibility_context"]["feasibility_mode"] = "measurement_only"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        p0.validate_resolution_evidence(cfg, context)
