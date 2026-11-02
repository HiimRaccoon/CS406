"""Frozen E1-HR inputs and immutable, allocation-free run declarations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/E1-HR/e1hr_numeric_contract_v1.json"
CONTRACT_SHA256 = "e23fa67e1f41c7bea43446e44961a215ca488aac72ae21f5e56395e097c0a35d"
G1_PATH = ROOT / "docs/E1-HR/hr_g1_approval.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError(f"Duplicate JSON key: {k}")
            obj[k] = v
        return obj
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def _frozen(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("Expected exact frozen E1-HR contract SHA-256")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.read_text().split() != [CONTRACT_SHA256, path.name]:
        raise ValueError("Missing or inconsistent contract sidecar")
    return read_json(path)


def require_approval(path: Path, gate: str, run_id: str | None = None) -> dict:
    record = read_json(path)
    if (record.get("gate") != gate or record.get("approved") is not True
            or record.get("contract_sha256") != CONTRACT_SHA256
            or record.get("experiment") != "E1-HR" or record.get("resolution_wh") != [1344, 576]
            or gate not in {"HR-G1", "HR-G2", "HR-G3a", "HR-G3b", "HR-TARGET-AUTH"}):
        raise ValueError(f"Missing reviewed {gate} approval for this contract")
    if run_id is not None and run_id not in record.get("run_ids", []):
        raise ValueError(f"Run is not authorized by {gate}")
    return record


def load_recipe(path: Path = CONTRACT_PATH) -> dict:
    recipe = _frozen(path)
    require_approval(G1_PATH, "HR-G1")
    return recipe


def validate_recipe(recipe: dict) -> dict:
    try:
        same = fingerprint(recipe) == fingerprint(load_recipe())
    except (TypeError, OverflowError) as exc:
        raise ValueError("Invalid E1-HR recipe") from exc
    if not same:
        raise ValueError("Recipe differs from the complete G1-frozen E1-HR contract; no defaults/overrides")
    return copy.deepcopy(recipe)


def resolve_run(recipe: dict, *, stage: str, lr: float, seed: int, paths: dict) -> dict:
    recipe = validate_recipe(recipe)
    campaign = recipe["campaign"]
    seeds = [campaign["tuning_seed"]] if stage == "tuning" else campaign["final_seeds"] if stage == "final" else []
    if type(seed) is not int or seed not in seeds:
        raise ValueError("Invalid stage or seed role")
    if type(lr) not in {int, float} or not math.isfinite(lr) or lr not in campaign["lr_candidates"]:
        raise ValueError("LR must be one of the four reviewed candidates")
    required = {"data_root", "weights", "runs_dir", "conformance_report", "acceptance_record", "authorization_record"}
    optional = {"conformance_report", "authorization_record", "acceptance_record", "selection"}
    if (not isinstance(paths, dict) or required - paths.keys() or paths.keys() - (required | optional)
            or any(not isinstance(p, str) or not p.strip() for p in paths.values())):
        raise ValueError("Explicit E1-HR paths required; unknown overrides rejected")
    if stage == "final" and "selection" not in paths:
        raise ValueError("Final declaration requires a locked selection reference")
    scientific = {"experiment": "E1-HR", "resolution_wh": [1344, 576], "budget": copy.deepcopy(recipe["budget"]),
                  "contract_sha256": CONTRACT_SHA256, "stage": stage, "lr": lr, "seed": seed}
    run_id = f"e1hr-{stage}-lr{lr:.8f}-seed{seed}"
    result = {"kind": "official_e1hr_run_config_v1", "recipe": recipe, **scientific,
              "run_id": run_id, "paths": copy.deepcopy(paths), "scientific_sha256": fingerprint(scientific)}
    if stage == "final":
        result["selection_sha256"] = sha256_file(Path(paths["selection"]))
    result["config_sha256"] = fingerprint(result)
    return result


def validate_run(config: dict) -> dict:
    try:
        expected = resolve_run(config["recipe"], stage=config["stage"], lr=config["lr"],
                               seed=config["seed"], paths=config["paths"])
        if fingerprint(config) != fingerprint(expected):
            raise ValueError("Changed or unknown run configuration fields")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete E1-HR run configuration") from exc
    return copy.deepcopy(config)


def scientific_identity(identity: dict) -> dict:
    """Equality projection; audit commits/tooling remain recorded, never compared."""
    if (not isinstance(identity, dict) or set(identity) != {"scientific_runtime", "audit"}
            or not isinstance(identity["scientific_runtime"], dict) or not identity["scientific_runtime"]
            or not isinstance(identity["audit"], dict)):
        raise ValueError("Explicit scientific/runtime identity and separate audit metadata required")
    fingerprint(identity["scientific_runtime"])
    return copy.deepcopy(identity["scientific_runtime"])
