"""Run one declared resolution/microbatch; no automatic candidate/search loop."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import fingerprint, load_config, parser, validate
from .dataset import FixedManifest, PilotDataset, build_loader
from .engine import make_scaler, synchronize, train_window
from .models import build_model, build_optimizer, pretrained_metadata
from .runtime import apply_runtime, create_run, environment, identity, run_lock, validate_gates, write_json
from ..preprocessing.geoguessr_v1 import PREPROCESS_SHA256, Preprocessor


def feasibility_context(cfg, manifest, weights, env):
    policy = cfg.get("resolution_policy", {})
    return {"version": 1, "protocol_id": cfg["protocol_id"],
            "ordered_candidates": policy.get("ordered_candidates"), "criteria_id": policy.get("criteria_id"),
            "feasibility": cfg["feasibility"], "seed": cfg["seed"],
            "runtime": cfg["runtime"], "initialization": cfg["initialization"],
            "batch_policy": policy.get("batch_policy"),
            "dataset_root": str(Path(cfg["paths"]["data_root"]).resolve()),
            "fingerprints": manifest.fingerprints, "preprocessing_sha256": PREPROCESS_SHA256,
            "pretrained_sha256": weights["sha256"], "identity": identity(manifest, weights, env),
            "gpu_total_bytes": env.get("gpu_total_bytes"), "driver": env.get("driver"),
            "budget": cfg["budget"], "optimizer": cfg["optimizer"],
            "augmentation": cfg["augmentation"], "clipping": cfg["clipping"]}


def evidence_fields(context):
    return {"feasibility_context": context,
            "feasibility_fingerprint": fingerprint(context) if context is not None else None}


def validate_resolution_evidence(cfg, context=None):
    if cfg["experiment"] != "e1hr":
        return
    policy = cfg["resolution_policy"]
    predecessors = policy["ordered_candidates"][:policy["ordered_candidates"].index(cfg["global_wh"])]
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in policy.get("prior_failure_reports", [])]
    if len(reports) != len(predecessors):
        raise ValueError("Every earlier HR candidate needs one ordered failure report; no skipping")
    for i, (wh, report) in enumerate(zip(predecessors, reports)):
        if (report.get("purpose") != "pilot" or report.get("stage") != "p0" or report.get("experiment") != "e1hr"
                or report.get("global_wh") != wh or report.get("status") != "fail"
                or report.get("failure_kind") not in {"oom", "criteria"}
                or report.get("resolution_policy", {}).get("criteria_id") != policy["criteria_id"]
                or report["resolution_policy"].get("ordered_candidates") != policy["ordered_candidates"]):
            raise ValueError("Fallback evidence must be a genuine earlier HR feasibility/OOM failure under the same policy")
        previous = report.get("feasibility_context")
        digest = report.get("feasibility_fingerprint")
        if (not context or not isinstance(previous, dict) or digest != fingerprint(previous)
                or digest != fingerprint(context) or previous != context):
            raise ValueError("HR feasibility context/fingerprint mismatch; prior failure cannot reject this resolution")
        batch_policy = policy["batch_policy"]
        if (report.get("batch") != batch_policy["candidate_batches"][i]
                or report.get("effective_batch") != batch_policy["effective_batch"]
                or report["resolution_policy"].get("batch_policy") != batch_policy):
            raise ValueError("HR feasibility failure used a batch outside the reviewed candidate policy")


def failure_summary(cfg, folder, exc):
    context_path = folder / "feasibility_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else None
    return {"purpose": "pilot", "stage": "p0", "experiment": cfg["experiment"], "status": "fail",
            "failure_kind": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "infrastructure_or_numerical",
            "protocol_id": cfg["protocol_id"], "global_wh": cfg["global_wh"],
            "resolution_policy": cfg.get("resolution_policy"), "batch": cfg["batch"],
            "effective_batch": cfg["batch"]["microbatch"] * cfg["batch"]["accumulation"],
            "error_type": type(exc).__name__, "error": str(exc), "automatic_fallback": False,
            **evidence_fields(context)}


def run(cfg, folder):
    validate(cfg)
    manifest = FixedManifest(cfg["artifacts_dir"])
    validate_gates(cfg, manifest)
    processor = Preprocessor(cfg["experiment"], cfg["global_wh"], cfg["preprocessing_config"])
    apply_runtime(cfg)
    # Persist verified identity before any model allocation that can OOM.
    weights = pretrained_metadata(cfg["paths"]["weights"], cfg["expected_weight_sha256"])
    env = environment(cfg)
    write_json(folder / "environment.json", env)
    context = feasibility_context(cfg, manifest, weights, env)
    write_json(folder / "feasibility_context.json", context)
    validate_resolution_evidence(cfg, context)
    device = torch.device("cuda")
    model, weights = build_model(cfg["experiment"], cfg["paths"]["weights"], cfg["initialization"],
                                cfg["expected_weight_sha256"])
    model.to(device)
    optimizer, groups = build_optimizer(model, cfg["optimizer"])
    write_json(folder / "model_metadata.json", {**weights, "optimizer_groups": groups,
                                               "parameters": sum(p.numel() for p in model.parameters())})
    forward_model = torch.compile(model) if cfg["runtime"]["torch_compile"] else model
    dataset = PilotDataset(manifest, Path(cfg["paths"]["data_root"]), "train",
                           processor,
                           cfg["seed"], cfg["augmentation"])
    loader = build_loader(dataset, cfg, train=True)
    scaler = make_scaler(cfg["runtime"])
    kwargs = dict(device=device, runtime=cfg["runtime"], accumulation=cfg["batch"]["accumulation"],
                  clipping=cfg["clipping"])
    warmup = None
    iterator = iter(loader)
    if cfg["budget"]["warmup_updates"]:
        warmup = train_window(forward_model, loader, optimizer, scaler,
                              max_updates=cfg["budget"]["warmup_updates"], iterator=iterator, **kwargs)
        if warmup["optimizer_updates"] != cfg["budget"]["warmup_updates"]:
            raise ValueError("Warmup window exceeds one available train epoch")
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured = train_window(forward_model, loader, optimizer, scaler,
                            max_updates=cfg["budget"]["measure_updates"], iterator=iterator, **kwargs)
    if measured["optimizer_updates"] != cfg["budget"]["measure_updates"]:
        raise ValueError("Measured window exceeds one available train epoch")
    memory = {"measured_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "measured_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    epoch = None
    if cfg["budget"]["full_epoch"]:
        loader.sampler.set_epoch(1)
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        epoch = train_window(forward_model, loader, optimizer, scaler, **kwargs)
        if epoch["samples"] != len(dataset):
            raise ValueError("Full-epoch measurement did not cover exact train membership")
        memory.update(epoch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      epoch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    criteria = cfg["feasibility"]
    reserved = max(v for k, v in memory.items() if "reserved" in k) / 2**30
    failures = []
    if reserved > criteria["max_peak_reserved_gib"]:
        failures.append("peak reserved VRAM exceeds declared threshold")
    if measured["images_per_second"] < criteria["min_images_per_second"]:
        failures.append("steady-state throughput below declared threshold")
    if epoch and epoch["seconds"] > criteria["max_epoch_seconds"]:
        failures.append("full-epoch walltime exceeds declared threshold")
    timing = {"benchmark_warmup": warmup, "steady_state": measured, "full_epoch": epoch,
              "scope": "end-to-end train batches including loader/transfer/numerical checks; excludes validation/checkpoint I/O"}
    write_json(folder / "timing.json", timing)
    write_json(folder / "memory.json", memory)
    return {"purpose": "pilot", "stage": "p0", "experiment": cfg["experiment"],
            "protocol_id": cfg["protocol_id"], **evidence_fields(context),
            "status": "fail" if failures else "pass", "failures": failures,
            "failure_kind": "criteria" if failures else None,
            "seed": cfg["seed"], "global_wh": cfg["global_wh"], "batch": cfg["batch"],
            "effective_batch": cfg["batch"]["microbatch"] * cfg["batch"]["accumulation"],
            "runtime_policy": cfg["runtime"], "full_epoch_measured": epoch is not None,
            "initialization_policy": cfg["initialization"],
            "resolution_policy": cfg.get("resolution_policy"),
            "fingerprints": manifest.fingerprints, "preprocessing_sha256": PREPROCESS_SHA256,
            "identity": identity(manifest, weights, env), "timing": timing, "memory": memory,
            "criteria": criteria, "model_quality_or_convergence_claim": False,
            "test_images_opened": 0}


def main():
    p = parser(__doc__)
    args = p.parse_args()
    cfg = load_config(args.config, data_root=args.data_root, weights=args.weights, runs_dir=args.runs_dir)
    if cfg["stage"] != "p0":
        p.error("Expected P0 config")
    if args.check_config:
        print("P0 config complete; no model/job run")
        return
    folder = create_run(cfg, args.run_id)
    with run_lock(folder):
        try:
            result = run(cfg, folder)
        except Exception as exc:
            result = failure_summary(cfg, folder, exc)
            write_json(folder / "run_summary.json", result)
            raise
        write_json(folder / "run_summary.json", result)
    print(folder)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
