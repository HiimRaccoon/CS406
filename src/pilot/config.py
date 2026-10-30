"""Pilot configuration: unresolved research values are errors, never defaults."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


def normalize_weight_sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value.strip().lower()):
        raise ConfigError("expected_weight_sha256 must be a reviewed full 64-character SHA-256")
    return value.strip().lower()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def repo_path(value):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else ROOT / p


def validate(cfg: dict):
    def get(key):
        value = cfg
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    if cfg.get("purpose") != "pilot" or cfg.get("stage") not in {"conformance", "p0", "p1"}:
        raise ConfigError("Only conformance/P0/P1 pilot configs are accepted")
    stage = cfg["stage"]
    required = ["paths.data_root", "preprocessing_config", "artifacts_dir"]
    if stage == "conformance":
        required += ["samples_per_layout", "output_dir"]
    else:
        required += ["protocol_id", "seed", "paths.weights", "paths.runs_dir", "global_wh", "expected_weight_sha256",
                     "batch.microbatch", "batch.accumulation", "initialization.method", "initialization.seed",
                     "runtime.device", "runtime.num_workers", "runtime.pin_memory", "runtime.persistent_workers",
                     "runtime.cpu_threads", "runtime.amp.enabled", "runtime.amp.dtype", "runtime.scaler.enabled",
                     "runtime.tf32_matmul", "runtime.tf32_cudnn", "runtime.matmul_precision",
                     "runtime.cudnn_benchmark", "runtime.cudnn_deterministic", "runtime.deterministic_algorithms",
                     "runtime.deterministic_warn_only", "runtime.torch_compile", "optimizer.lr",
                     "optimizer.betas", "optimizer.eps", "optimizer.weight_decay", "augmentation.enabled",
                     "clipping.enabled", "gates.conformance_report"]
        if stage == "p0":
            required += ["budget.warmup_updates", "budget.measure_updates", "budget.full_epoch",
                         "feasibility.max_peak_reserved_gib", "feasibility.min_images_per_second"]
            if get("budget.full_epoch"):
                required += ["feasibility.max_epoch_seconds"]
        else:
            required += ["budget.max_epochs", "scheduler.warmup_updates", "scheduler.start_factor",
                         "scheduler.min_lr", "checkpoint.tolerance", "checkpoint.tie_break", "gates.p0_summaries"]

    if stage != "conformance":
        if cfg.get("experiment") not in {"e1", "e1hr", "e2"}:
            raise ConfigError("Only E1/E1-HR/E2 pilot model profiles are accepted")
        if get("runtime.num_workers"):
            required += ["runtime.prefetch_factor"]
        if get("runtime.scaler.enabled"):
            required += ["runtime.scaler.init_scale", "runtime.scaler.growth_factor",
                         "runtime.scaler.backoff_factor", "runtime.scaler.growth_interval"]
        if get("augmentation.enabled"):
            required += ["augmentation.brightness", "augmentation.contrast"]
        if get("clipping.enabled"):
            required += ["clipping.max_norm"]
        if get("initialization.method") == "trunc_normal":
            required += ["initialization.std", "initialization.bias"]
    missing = [key for key in required if get(key) is None]
    if missing:
        raise ConfigError("Pending values must be reviewed/filled before this command runs:\n  " + "\n  ".join(missing))
    fingerprint(cfg)  # rejects NaN/Infinity
    if stage == "conformance":
        if type(cfg["samples_per_layout"]) is not int or cfg["samples_per_layout"] < 1:
            raise ConfigError("samples_per_layout must be a positive integer")
        return cfg
    cfg["expected_weight_sha256"] = normalize_weight_sha256(cfg["expected_weight_sha256"])
    for key in ["seed", "initialization.seed", "batch.microbatch", "batch.accumulation",
                "runtime.num_workers", "runtime.cpu_threads"]:
        value = get(key)
        if type(value) is not int or value < (0 if key in {"seed", "initialization.seed", "runtime.num_workers"} else 1):
            raise ConfigError(f"Invalid integer: {key}")
    for key in ["runtime.pin_memory", "runtime.persistent_workers", "runtime.amp.enabled",
                "runtime.scaler.enabled", "runtime.tf32_matmul", "runtime.tf32_cudnn",
                "runtime.cudnn_benchmark", "runtime.cudnn_deterministic", "runtime.deterministic_algorithms",
                "runtime.deterministic_warn_only", "runtime.torch_compile", "augmentation.enabled", "clipping.enabled"]:
        if type(get(key)) is not bool:
            raise ConfigError(f"Explicit boolean required: {key}")
    if cfg["runtime"]["persistent_workers"] and not cfg["runtime"]["num_workers"]:
        raise ConfigError("persistent_workers requires num_workers > 0")
    if cfg["runtime"]["num_workers"] and (type(get("runtime.prefetch_factor")) is not int or get("runtime.prefetch_factor") < 1):
        raise ConfigError("Positive prefetch_factor required with workers")
    if cfg["runtime"]["device"] != "cuda":
        raise ConfigError("P0/P1 runners require CUDA; CPU tests exercise the mechanics separately")
    if cfg["runtime"]["amp"]["dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ConfigError("Unknown AMP dtype")
    if cfg["runtime"]["amp"]["enabled"] and cfg["runtime"]["amp"]["dtype"] == "float32":
        raise ConfigError("Enabled CUDA autocast requires float16 or bfloat16")
    if cfg["runtime"]["scaler"]["enabled"] and not cfg["runtime"]["amp"]["enabled"]:
        raise ConfigError("GradScaler requires AMP in this pilot")
    if cfg["runtime"]["matmul_precision"] not in {"highest", "high", "medium"}:
        raise ConfigError("Explicit float32 matmul precision required")
    if cfg["runtime"]["scaler"]["enabled"]:
        scaler = cfg["runtime"]["scaler"]
        if (scaler["init_scale"] <= 0 or scaler["growth_factor"] <= 1
                or not 0 < scaler["backoff_factor"] < 1
                or type(scaler["growth_interval"]) is not int or scaler["growth_interval"] < 1):
            raise ConfigError("Invalid explicit GradScaler parameters")
    wh = cfg["global_wh"]
    if not isinstance(wh, list) or len(wh) != 2 or any(type(x) is not int or x < 32 for x in wh):
        raise ConfigError("global_wh must contain [width,height] integers >=32")
    if cfg["experiment"] != "e1hr" and wh != [896, 384]:
        raise ConfigError("E1/E2 global resolution is frozen")
    if cfg["initialization"]["method"] not in {"linear_default", "trunc_normal"}:
        raise ConfigError("Select an explicit supported classifier initialization")
    if cfg["initialization"]["method"] == "trunc_normal" and cfg["initialization"]["std"] <= 0:
        raise ConfigError("Initialization std must be positive")
    opt = cfg["optimizer"]
    if opt["weight_decay"] != .05 or opt["lr"] <= 0 or opt["eps"] <= 0:
        raise ConfigError("Pilot requires AdamW, weight_decay=0.05, positive LR/eps")
    if len(opt["betas"]) != 2 or any(not 0 <= x < 1 for x in opt["betas"]):
        raise ConfigError("Invalid AdamW betas")
    if any(k in cfg for k in ["class_weights", "class_weighting", "lr_grid", "test_split", "oversampling"]):
        raise ConfigError("Search, test access, and class weighting are outside this harness")
    if cfg["augmentation"]["enabled"]:
        if any(not 0 <= cfg["augmentation"][key] <= 1 for key in ("brightness", "contrast")):
            raise ConfigError("Jitter amplitudes must be in [0,1]")
    if cfg["clipping"]["enabled"] and cfg["clipping"]["max_norm"] <= 0:
        raise ConfigError("Invalid clipping threshold")
    budget = cfg["budget"]
    if stage == "p0":
        if any(cfg["feasibility"][k] <= 0 for k in ("max_peak_reserved_gib", "min_images_per_second")):
            raise ConfigError("P0 feasibility thresholds must be positive")
        if budget["full_epoch"] and cfg["feasibility"]["max_epoch_seconds"] <= 0:
            raise ConfigError("Positive full-epoch walltime threshold required")
        if type(budget["full_epoch"]) is not bool:
            raise ConfigError("full_epoch must be explicit")
        for key in ("warmup_updates", "measure_updates"):
            if type(budget[key]) is not int or budget[key] < (0 if key == "warmup_updates" else 1):
                raise ConfigError("Invalid P0 bounded measurement window")
    else:
        if type(budget["max_epochs"]) is not int or budget["max_epochs"] < 1:
            raise ConfigError("P1 needs a finite positive max_epochs")
        sched = cfg["scheduler"]
        if type(sched["warmup_updates"]) is not int or sched["warmup_updates"] < 0:
            raise ConfigError("Invalid warmup_updates")
        if not 0 < sched["start_factor"] <= 1 or not 0 <= sched["min_lr"] <= opt["lr"]:
            raise ConfigError("Invalid warmup/cosine parameters")
        if cfg["checkpoint"]["tolerance"] < 0 or cfg["checkpoint"]["tie_break"] not in {"earlier", "nll"}:
            raise ConfigError("Explicit checkpoint tolerance/tie-break required")
    if cfg["experiment"] == "e1hr":
        policy = cfg.get("resolution_policy", {})
        if wh not in policy.get("ordered_candidates", []) or not policy.get("criteria_id"):
            raise ConfigError("E1-HR needs reviewed ordered_candidates and criteria_id; runner never loops resolutions")
        if policy["ordered_candidates"].index(wh) > 0 and not policy.get("prior_failure_reports"):
            raise ConfigError("Fallback candidate requires explicit prior failure reports")
        batch_policy = policy.get("batch_policy", {})
        effective = batch_policy.get("effective_batch")
        batches = batch_policy.get("candidate_batches", [])
        if type(effective) is not int or effective < 1 or len(batches) != len(policy["ordered_candidates"]):
            raise ConfigError("E1-HR needs a reviewed batch_policy: effective_batch and one batch per ordered candidate")
        for batch in batches:
            if (not isinstance(batch, dict) or set(batch) != {"microbatch", "accumulation"}
                    or any(type(batch[k]) is not int or batch[k] < 1 for k in batch)
                    or batch["microbatch"] * batch["accumulation"] != effective):
                raise ConfigError("Reviewed HR candidate batches must have the same positive effective batch")
        if cfg["batch"] != batches[policy["ordered_candidates"].index(wh)]:
            raise ConfigError("Actual HR batch is outside the reviewed candidate batch policy")
    return cfg


def load_config(path, *, data_root=None, weights=None, runs_dir=None):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    paths = cfg.setdefault("paths", {})
    for key, cli, env in [("data_root", data_root, "CS406_DATA_ROOT"),
                          ("weights", weights, "CS406_WEIGHTS_PATH"), ("runs_dir", runs_dir, "CS406_RUNS_DIR")]:
        value = cli or os.environ.get(env) or paths.get(key)
        if value:
            paths[key] = str(repo_path(value).resolve())
    validate(cfg)
    for key in ("preprocessing_config", "artifacts_dir", "output_dir"):
        if cfg.get(key):
            cfg[key] = str(repo_path(cfg[key]).resolve())
    for key, value in cfg.get("gates", {}).items():
        if value:
            cfg["gates"][key] = ([str(repo_path(v).resolve()) for v in value]
                                 if isinstance(value, list) else str(repo_path(value).resolve()))
    policy = cfg.get("resolution_policy", {})
    if policy.get("prior_failure_reports"):
        policy["prior_failure_reports"] = [str(repo_path(p).resolve()) for p in policy["prior_failure_reports"]]
    return cfg


def parser(description, *, resume=False):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--data-root")
    p.add_argument("--weights")
    p.add_argument("--runs-dir")
    p.add_argument("--run-id")
    p.add_argument("--check-config", action="store_true")
    if resume:
        p.add_argument("--resume", type=Path)
    return p
