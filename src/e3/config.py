"""Strict records and permissions; imports allocate no model or GPU."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
NUMERIC_SHA256 = "08e042448f2f4306b8be8bbd3fc0e7466641757b8f023fbd79a463c09ad13c0e"
CACHE_SHA256 = "1626caa8bf87900ebcbd6962bcfed846d6a8c301a83d922162d07be7c9cf33bc"
SCHEMA_SHA256 = "2146f541a46511826779914a794c85b1adca4a8c95826fc311c0a4301a0baba6"
SOURCE_SHA256 = "5f3fa03c908cccb920f2135ab594050a102211aef950885ac3c26f00366093df"
G1_PATH = ROOT / f"artifacts/E3/approvals/v1/E3-G1/{NUMERIC_SHA256}.json"
DIMENSIONS = {"E1": 768, "E2": 1536}
SEEDS = (2001, 3001, 4001)
OP_GATES = {"bounded_acceptance": "E3-TARGET-AUTH", "bulk_cache": "E3-CACHE-AUTH",
            "pilot_head": "E3-PILOT-AUTH", "final_head": "E3-FINAL-AUTH"}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON number")
    return parsed


def parse_json(text: str):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON constant: {value}")
    try:
        return json.loads(text, object_pairs_hook=_pairs,
                          parse_float=_float, parse_constant=invalid)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid strict JSON") from exc


def read_json(path: Path) -> dict:
    return parse_json(Path(path).read_bytes().decode("utf-8"))


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if (not relative or posix.is_absolute() or ".." in posix.parts or ":" in relative
            or "\\" in relative or posix.as_posix() != relative):
        raise ValueError("Unsafe relative path")
    base = Path(root).resolve()
    result = (base / relative).resolve()
    if not result.is_relative_to(base):
        raise ValueError("Path/symlink escapes approved root")
    return result


def verify_payload(binding: dict, root: Path = ROOT) -> Path:
    if set(binding) != {"relative_path", "bytes", "sha256"}:
        raise ValueError("Incomplete payload binding")
    path = contained(root, binding["relative_path"])
    if (not path.is_file() or path.stat().st_size != binding["bytes"]
            or sha256_file(path) != binding["sha256"]):
        raise ValueError(f"Payload bytes/hash mismatch: {path}")
    return path


def payload(path: Path, root: Path = ROOT) -> dict:
    path = Path(path).resolve()
    relative = path.relative_to(Path(root).resolve()).as_posix()
    contained(root, relative)
    return {"relative_path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _pinned(relative, digest):
    path = ROOT / relative
    if sha256_file(path) != digest:
        raise ValueError(f"Approved policy bytes changed: {relative}")
    return read_json(path)


def policy() -> dict:
    return _pinned("docs/E3/e3_cache_pilot_contract_v1.json", CACHE_SHA256)


def registry() -> dict:
    return _pinned("docs/E3/e3_source_manifest_v1.json", SOURCE_SHA256)


def numeric() -> dict:
    return _pinned("docs/E3/e3_numeric_contract_v1.json", NUMERIC_SHA256)


def schemas() -> dict:
    return _pinned("docs/E3/schemas/e3_record_schemas_v1.json", SCHEMA_SHA256)


def validate_record(value: dict, definition: str | None = None) -> dict:
    bundle = schemas()
    schema = {"$ref": f"#/$defs/{definition}", "$defs": bundle["$defs"]} if definition else bundle
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise ValueError(f"Invalid E3 record: {errors[0].message}")
    return value


def validate_common_protocol(protocol: dict) -> dict:
    validate_record(protocol, "pilot_protocol")
    if not any(protocol == p for p in policy()["candidate_profiles"]):
        raise ValueError("Exact common profile required; no family/arm/seed overrides")
    core = {k: v for k, v in protocol.items() if k != "candidate_profile_sha256"}
    if fingerprint(core) != protocol["candidate_profile_sha256"]:
        raise ValueError("Common profile fingerprint mismatch")
    return copy.deepcopy(protocol)


def require_approval(path: Path, *, gate: str, expected: dict) -> dict:
    receipt = validate_record(read_json(path), "gate_receipt_schema_only")
    if receipt["gate"] != gate:
        raise ValueError("Wrong stage receipt")
    evidence = contained(ROOT, receipt["approval_evidence_path"])
    if sha256_file(evidence) != receipt["approval_evidence_sha256"]:
        raise ValueError("Approval evidence hash mismatch")
    decision = read_json(evidence)
    if (decision.get("kind") != "e3_user_gate_decision_v1" or decision.get("gate") != gate
            or decision.get("status") != "APPROVED"
            or any(decision.get(k) != receipt[k] for k in
                   ("approved_subjects", "allowed_operations", "allowed_splits"))):
        raise ValueError("Matching explicit user decision required")
    subjects = {r["role"]: r["sha256"] for r in receipt["approved_subjects"]}
    if len(subjects) != len(receipt["approved_subjects"]):
        raise ValueError("Duplicate approved subject roles")
    if any(subjects.get(k) != v for k, v in expected.get("subjects", {}).items()):
        raise ValueError("Unapproved subject bytes")
    if expected.get("operation") and expected["operation"] not in receipt["allowed_operations"]:
        raise ValueError("Operation outside approval")
    if expected.get("split") and expected["split"] not in receipt["allowed_splits"]:
        raise ValueError("Split outside approval")
    return receipt


def g1_receipt() -> dict:
    return require_approval(G1_PATH, gate="E3-G1", expected={"operation": "implementation", "subjects": {
        "docs/E3/e3_numeric_contract_v1.json": NUMERIC_SHA256,
        "docs/E3/e3_cache_pilot_contract_v1.json": CACHE_SHA256,
        "docs/E3/schemas/e3_record_schemas_v1.json": SCHEMA_SHA256,
        "docs/E3/e3_source_manifest_v1.json": SOURCE_SHA256}})


def reject_unresolved(value):
    if value is None or (isinstance(value, str) and value.upper() in {"PENDING", "USE DEFAULT"}):
        raise ValueError("Unresolved executable input")
    if isinstance(value, dict):
        for item in value.values():
            reject_unresolved(item)
    elif isinstance(value, list):
        for item in value:
            reject_unresolved(item)


def load_contract(path: Path, *, stage: str) -> dict:
    value = read_json(path)
    if stage == "design":
        if value.get("kind") == "e3_numeric_contract_v1":
            if sha256_file(path) != NUMERIC_SHA256:
                raise ValueError("Numeric candidate differs from G1 bytes")
            return value
        return validate_record(value)
    kinds = {"bulk_cache": "e3_cache_request_v1", "bounded_acceptance": "e3_acceptance_request_v1",
             "pilot_head": "e3_pilot_run_v1", "final_head": "e3_final_head_run_v1"}
    if value.get("kind") != kinds.get(stage):
        raise ValueError("Candidate/design kind is not an executable stage declaration")
    validate_record(value)
    reject_unresolved(value)
    return value


def validate_source_binding(source: dict, *, accepted: bool = True) -> dict:
    validate_record(source, "source_binding" if accepted else "source_input")
    if Path(source["actual_binary_path"]).suffix != ".pt" or Path(source["actual_report_path"]).suffix != ".json":
        raise ValueError("Source inputs must have checkpoint/report roles, never image/feature roles")
    reference = next((r for r in registry()["selected_sources"] if (r["family"], r["source_seed"]) ==
                      (source["family"], source["source_seed"])), None)
    mapping = {"selected_epoch": "selected_epoch", "binary_sha256": "required_binary_sha256",
               "report_sha256": "required_report_sha256", "source_contract_sha256": "required_source_contract_sha256"}
    if reference is None or any(source[k] != reference[v] for k, v in mapping.items()):
        raise ValueError("Selected source lineage differs from fixed registry")
    if source["source_manifest_sha256"] != SOURCE_SHA256:
        raise ValueError("Source manifest identity mismatch")
    return source


def source_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else contained(ROOT, value)


def resolve_cache_request(contract: dict, source: dict, paths: dict) -> dict:
    if contract != policy():
        raise ValueError("Cache candidate differs from G1 policy")
    validate_source_binding(source)
    if paths["split"] not in {"train", "val"}:
        raise ValueError("Test access forbidden")
    value = {"kind": "e3_cache_request_v1", "family": source["family"], "source_seed": source["source_seed"],
             "split": paths["split"], "source": source, "runtime": paths["runtime"],
             "fixed_artifact_sha256": {k: v["sha256"] for k, v in numeric()["data"]["artifacts"].items()},
             "preprocessing_sha256": numeric()["preprocessing"]["config"]["sha256"],
             "cache_policy_sha256": fingerprint(contract["storage"]), "output_root": paths["output_root"],
             "required_authorization_gate": "E3-CACHE-AUTH"}
    validate_record(value); reject_unresolved(value)
    return value


def resolve_head_run(contract: dict, protocol: dict, *, stage: str, family: str,
                     seed: int, arm: str, bindings: dict) -> dict:
    validate_common_protocol(protocol)
    if (contract != policy() or stage not in {"pilot", "final"} or family not in DIMENSIONS
            or seed not in SEEDS or arm not in {"natural", "crt"}):
        raise ValueError("Invalid head run scope")
    if stage == "pilot" and seed != 2001:
        raise ValueError("Pilot source seed must be 2001")
    if stage == "final":
        if not bindings.get("head_freeze_receipt"):
            raise ValueError("Final profile requires HEAD-FREEZE")
        require_approval(bindings["head_freeze_receipt"], gate="E3-HEAD-FREEZE",
                         expected={"subjects": {"common_profile": protocol["candidate_profile_sha256"]}})
    required = {"source", "cache", "initialization", "runtime"}
    if not required.issubset(bindings):
        raise ValueError("Missing source/cache/init/runtime bindings")
    validate_source_binding(bindings["source"])
    if (bindings["source"]["family"], bindings["source"]["source_seed"]) != (family, seed):
        raise ValueError("Mismatched source family/seed")
    if bindings["initialization"]["fan_in"] != DIMENSIONS[family]:
        raise ValueError("Head dimension mismatch")
    value = {"kind": "e3_pilot_run_v1" if stage == "pilot" else "e3_final_head_run_v1",
             "stage": stage, "family": family, "source_seed": seed, "root_seed": 1001 if stage == "pilot" else seed,
             "arm": arm, **{k: copy.deepcopy(bindings[k]) for k in required},
             "profile": {"candidate_id": protocol["candidate_id"], "profile_sha256": protocol["candidate_profile_sha256"],
                         "numeric_contract_sha256": NUMERIC_SHA256, "cache_pilot_contract_sha256": CACHE_SHA256,
                         "schema_bundle_sha256": SCHEMA_SHA256},
             "original_budget": copy.deepcopy(protocol["budget"]),
             "required_authorization_gate": "E3-PILOT-AUTH" if stage == "pilot" else "E3-FINAL-AUTH"}
    if stage == "final":
        value["head_freeze_receipt_sha256"] = sha256_file(bindings["head_freeze_receipt"])
    value["run_id"] = f"e3-{stage}-{fingerprint(value)[:24]}"
    validate_record(value); reject_unresolved(value)
    return value
