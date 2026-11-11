import copy
import json

import pytest


def test_strict_json_and_canonical_identity(tmp_path):
    from src.e3.config import read_json, fingerprint
    p = tmp_path / "bad.json"
    for text in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1} trailing', '{"x":1e999}']:
        p.write_text(text)
        with pytest.raises(ValueError):
            read_json(p)
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})


def test_common_protocol_has_no_family_override():
    from src.e3.config import policy, validate_common_protocol
    p = copy.deepcopy(policy()["candidate_profiles"][0])
    assert validate_common_protocol(p) == p
    for change in [dict(family="E1"), dict(base_lr=0.01), dict(runtime={"amp": True})]:
        q = dict(p, **change)
        with pytest.raises(ValueError):
            validate_common_protocol(q)


def test_final_rejects_pending_or_unfrozen_profile(tmp_path):
    from src.e3.config import ROOT, load_contract, validate_record, resolve_head_run, policy
    with pytest.raises(ValueError):
        load_contract(ROOT / "docs/E3/e3_numeric_contract_v1.json", stage="final_head")
    with pytest.raises(ValueError):
        validate_record({"kind": "e3_final_head_run_v1", "profile": None})
    with pytest.raises(ValueError):
        resolve_head_run(policy(), policy()["candidate_profiles"][0], stage="final",
                         family="E1", seed=2001, arm="natural", bindings={})


def test_wrong_stage_receipt_fails_before_allocation(tmp_path):
    from src.e3.config import require_approval
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"kind": "e3_gate_receipt_v1", "gate": "E3-G1", "status": "APPROVED"}))
    with pytest.raises(ValueError):
        require_approval(receipt, gate="E3-FINAL-AUTH", expected={})


def test_approved_candidate_schema_and_g1_receipt():
    from src.e3.config import policy, registry, validate_record, g1_receipt
    validate_record(policy())
    validate_record(registry())
    assert g1_receipt()["allowed_splits"] == []
