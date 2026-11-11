"""Synthetic declarations have no operation receipts and cannot execute publicly."""
import hashlib
from src.e3.config import policy,registry,SOURCE_SHA256,resolve_head_run


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def declaration(family="E1",arm="natural"):
    ref=next(r for r in registry()["selected_sources"] if r["family"]==family and r["source_seed"]==2001)
    source={"family":family,"source_seed":2001,"selected_epoch":ref["selected_epoch"],
            "actual_binary_path":"nonofficial/no-binary.pt","binary_sha256":ref["required_binary_sha256"],
            "actual_report_path":"nonofficial/no-report.json","report_sha256":ref["required_report_sha256"],
            "source_contract_sha256":ref["required_source_contract_sha256"],"source_manifest_sha256":SOURCE_SHA256,
            "source_acceptance_sha256":digest("nonofficial acceptance observation")}
    return resolve_head_run(policy(),policy()["candidate_profiles"][0],stage="pilot",family=family,seed=2001,arm=arm,
        bindings={"source":source,"cache":{"train_index_path":"nonofficial/train/index.json","train_index_sha256":digest("train"),
        "validation_index_path":"nonofficial/val/index.json","validation_index_sha256":digest("val"),"cache_acceptance_sha256":digest("cache")},
        "initialization":{"tensor_payload_path":"nonofficial/init.pt","tensor_state_sha256":digest("init"),
        "initializer_policy_sha256":digest("initializer"),"initialization_seed":1,"fan_in":768 if family=="E1" else 1536},
        "runtime":{k:digest(k) for k in ["accepted_runtime_sha256","consumed_source_sha256","current_data_audit_sha256","actual_environment_evidence_sha256"]}})
