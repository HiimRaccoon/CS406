"""Small pilot-only runtime, metadata, artifact, and gate helpers."""
from __future__ import annotations

import contextlib
import datetime as dt
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from .config import ROOT, fingerprint
from ..preprocessing.geoguessr_v1 import PREPROCESS_SHA256, sha256_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_runtime(cfg):
    r = cfg["runtime"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use CPU unit tests/conformance locally, not fake P0 GPU measurements")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"] % 2**32)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(r["cpu_threads"])
    allocated_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus and r["cpu_threads"] > int(allocated_cpus):
        raise ValueError("Declared CPU thread count exceeds the Slurm allocation")
    torch.backends.cuda.matmul.allow_tf32 = r["tf32_matmul"]
    torch.set_float32_matmul_precision(r["matmul_precision"])
    torch.backends.cudnn.allow_tf32 = r["tf32_cudnn"]
    if torch.backends.cuda.matmul.allow_tf32 != r["tf32_matmul"]:
        raise ValueError("Incompatible TF32/matmul_precision settings; declare a consistent policy")
    torch.backends.cudnn.benchmark = r["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = r["cudnn_deterministic"]
    torch.use_deterministic_algorithms(r["deterministic_algorithms"], warn_only=r["deterministic_warn_only"])
    if r["amp"]["enabled"] and r["amp"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support requested bfloat16")


def environment(cfg=None):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for name in ("torch", "torchvision", "Pillow", "numpy", "pandas", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
              "git_commit": commit, "git_dirty": dirty, "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "gpu_name": None,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "allocation": {key: os.environ.get(key) for key in
                             ("SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_TIMELIMIT")},
              "runtime_environment": {key: os.environ.get(key) for key in
                                      ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS",
                                       "MKL_NUM_THREADS", "PYTHONHASHSEED")},
              "actual_torch_settings": {"cpu_threads": torch.get_num_threads(),
                                        "matmul_precision": torch.get_float32_matmul_precision(),
                                        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
              "runtime_policy": cfg["runtime"] if cfg else None}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        result.update(gpu_name=props.name, gpu_total_bytes=props.total_memory)
        try:
            result["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            result["driver"] = None
    return result


def create_run(cfg, run_id=None, resume=None):
    root = Path(cfg["paths"]["runs_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if resume:
        resume = Path(resume).resolve()
        folder = resume.parent
        if not folder.is_relative_to(root) or not resume.is_file():
            raise ValueError("Resume checkpoint must be an existing pilot run under configured runs_dir")
        if run_id and run_id != folder.name:
            raise ValueError("Resume run ID mismatch")
        original = json.loads((folder / "config_resolved.json").read_text(encoding="utf-8"))
        if fingerprint(original) != fingerprint(cfg):
            raise ValueError("Cannot change planned config, seed, or budget on resume")
        return folder
    run_id = run_id or f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}_{cfg['stage']}_{cfg['experiment']}_{uuid.uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) or run_id in {".", ".."}:
        raise ValueError("Unsafe run ID")
    folder = root / run_id
    folder.mkdir(exist_ok=False)
    write_json(folder / "config_resolved.json", cfg)
    write_json(folder / "run_summary.json", {"purpose": "pilot", "stage": cfg["stage"],
               "experiment": cfg["experiment"], "status": "created_not_completed",
               "seed": cfg["seed"], "note": "If no terminal summary follows, inspect logs for abrupt termination"})
    return folder


@contextlib.contextmanager
def run_lock(folder):
    with (Path(folder) / ".run.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def validate_gates(cfg, manifest):
    report = json.loads(Path(cfg["gates"]["conformance_report"]).read_text(encoding="utf-8"))
    if (report.get("status") != "pass" or report.get("preprocessing_sha256") != PREPROCESS_SHA256
            or report.get("fingerprints") != manifest.fingerprints or report.get("test_images_opened") != 0):
        raise ValueError("Missing/incompatible passing preprocessing conformance report")
    if report.get("image_hash_audit") != {"train": 34997, "val": 7497}:
        raise ValueError("Before GPU pilot, run conformance with --verify-all-train-val-hashes (never test images)")
    if report.get("dataset_root") != str(Path(cfg["paths"]["data_root"]).resolve()):
        raise ValueError("Conformance audit is for a different dataset root")
    if report.get("environment", {}).get("packages") != environment()["packages"]:
        raise ValueError("Rerun conformance in this exact target package environment")
    if cfg["stage"] == "p1":
        paths = cfg["gates"]["p0_summaries"]
        if not isinstance(paths, list) or len(paths) != 3:
            raise ValueError("P1 requires the three reviewed passing P0 summaries")
        summaries = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
        if {s.get("experiment") for s in summaries} != {"e1", "e1hr", "e2"}:
            raise ValueError("P0 gate must include E1, E1-HR, E2")
        effective = cfg["batch"]["microbatch"] * cfg["batch"]["accumulation"]
        for s in summaries:
            if (s.get("stage") != "p0" or s.get("status") != "pass" or not s.get("full_epoch_measured")
                    or s.get("fingerprints") != manifest.fingerprints
                    or s.get("preprocessing_sha256") != PREPROCESS_SHA256
                    or s.get("effective_batch") != effective or s.get("runtime_policy") != cfg["runtime"]
                    or s.get("seed") != cfg["seed"] or s.get("initialization_policy") != cfg["initialization"]):
                raise ValueError("P0 gate mismatch: status/full epoch/batch/runtime/seed/artifacts")
        own = next(s for s in summaries if s["experiment"] == cfg["experiment"])
        if own["global_wh"] != cfg["global_wh"] or own["batch"] != cfg["batch"]:
            raise ValueError("P1 resolution/microbatch differs from passing P0 configuration")
        return summaries
    return []


def identity(manifest, weights, env):
    sources = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted((ROOT / "src").rglob("*.py"))}
    return {"fingerprints": manifest.fingerprints, "preprocessing_sha256": PREPROCESS_SHA256,
            "pretrained_sha256": weights["sha256"], "packages": env["packages"],
            "cuda": env["cuda"], "cudnn": env["cudnn"], "gpu_name": env["gpu_name"],
            "git_commit": env["git_commit"], "git_dirty": env["git_dirty"],
            "runtime_environment": env["runtime_environment"],
            "source_sha256": fingerprint(sources)}
