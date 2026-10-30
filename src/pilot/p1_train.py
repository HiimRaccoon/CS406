"""Bounded P1 diagnostic training only; no official LR grid, final runs, or E3."""
from __future__ import annotations

import csv
import io
import math
import signal
import time
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config, parser
from .dataset import FixedManifest, PilotDataset, build_loader
from .engine import WarmupCosine, is_better, make_scaler, train_window, validate_epoch
from .models import build_model, build_optimizer
from .runtime import apply_runtime, create_run, environment, identity, run_lock, validate_gates, write_json
from ..preprocessing.geoguessr_v1 import Preprocessor


def write_history(folder, history):
    write_json(folder / "history.json", history)
    fields = ["epoch", "train_loss", "validation_nll", "top1", "top3", "top5", "macro_f1_supported",
              "weighted_f1", "epoch_seconds", "optimizer_updates", "lr_first", "lr_last"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in history:
        writer.writerow({"epoch": row["epoch"], "train_loss": row["train"]["loss"],
                         "validation_nll": row["validation"]["nll"],
                         **{k: row["validation"][k] for k in fields[3:8]},
                         "epoch_seconds": row["epoch_seconds"],
                         **{k: row["train"][k] for k in fields[9:]}})
    (folder / "metrics.csv").write_text(output.getvalue(), encoding="utf-8")
    # Standalone diagnostic SVG, no plotting-service/dependency or final-result claims.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="320">',
           '<rect width="800" height="320" fill="white"/>',
           '<text x="20" y="20">BOUNDED PILOT ONLY: train/val loss (left), validation Macro-F1 (right)</text>']
    for offset, curves in [(30, [("red", [r["train"]["loss"] for r in history]),
                                 ("blue", [r["validation"]["nll"] for r in history])]),
                           (420, [("green", [r["validation"]["macro_f1_supported"] for r in history])])]:
        maximum = max([v for _, values in curves for v in values] + [1e-9])
        svg.append(f'<path d="M{offset},45 V285 H{offset+340}" fill="none" stroke="black"/>')
        for color, values in curves:
            points = " ".join(f"{offset+i*340/max(1,len(values)-1):.2f},{285-v/maximum*220:.2f}" for i, v in enumerate(values))
            svg.append(f'<polyline points="{points}" stroke="{color}" fill="none"/>')
        svg.append(f'<text x="{offset}" y="310">epoch 1 .. {len(history)}; ymax={maximum:.4g}</text>')
    (folder / "learning_curves.svg").write_text("\n".join(svg) + "\n</svg>\n", encoding="utf-8")


def run(cfg, folder, resume=None, stop_requested=lambda: False):
    manifest = FixedManifest(cfg["artifacts_dir"])
    p0_summaries = validate_gates(cfg, manifest)
    apply_runtime(cfg)
    device = torch.device("cuda")
    model, weights = build_model(cfg["experiment"], cfg["paths"]["weights"], cfg["initialization"],
                                cfg["expected_weight_sha256"])
    model.to(device)
    optimizer, groups = build_optimizer(model, cfg["optimizer"])
    env = environment(cfg)
    run_identity = identity(manifest, weights, env)
    if any(s["identity"] != run_identity for s in p0_summaries):
        raise ValueError("P0 and P1 pretrained/data/preprocessing/package/GPU/source identities differ")
    write_json(folder / "environment.json", env)
    write_json(folder / "model_metadata.json", {**weights, "optimizer_groups": groups})
    processor = Preprocessor(cfg["experiment"], cfg["global_wh"], cfg["preprocessing_config"])
    train_data = PilotDataset(manifest, Path(cfg["paths"]["data_root"]), "train", processor,
                             cfg["seed"], cfg["augmentation"])
    val_data = PilotDataset(manifest, Path(cfg["paths"]["data_root"]), "val", processor, cfg["seed"])
    train_loader, val_loader = build_loader(train_data, cfg, train=True), build_loader(val_data, cfg, train=False)
    updates_per_epoch = math.ceil(len(train_loader) / cfg["batch"]["accumulation"])
    total_updates = updates_per_epoch * cfg["budget"]["max_epochs"]
    scheduler = WarmupCosine(optimizer, total_updates, cfg["scheduler"]["warmup_updates"],
                             cfg["scheduler"]["start_factor"], cfg["scheduler"]["min_lr"])
    scaler = make_scaler(cfg["runtime"])
    generators = {"train": train_loader.generator, "val": val_loader.generator}
    history, best, completed_epochs, updates = [], None, 0, 0
    checkpoint_args = dict(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                           cfg=cfg, loader_generators=generators, identity=run_identity)
    if resume:
        state = load_checkpoint(resume, **checkpoint_args)
        history, best = state["history"], state["best"]
        completed_epochs, updates = state["epoch"], state["global_optimizer_update"]
        if not 0 <= completed_epochs <= cfg["budget"]["max_epochs"] or updates != completed_epochs * updates_per_epoch:
            raise ValueError("Resume checkpoint is not a valid epoch-boundary prefix")
    else:
        save_checkpoint(folder / "checkpoint_last.pt", epoch=0, updates=0, history=[], best=None, **checkpoint_args)
    forward_model = torch.compile(model) if cfg["runtime"]["torch_compile"] else model
    for epoch in range(completed_epochs, cfg["budget"]["max_epochs"]):
        if stop_requested():
            break
        train_loader.sampler.set_epoch(epoch)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        train = train_window(forward_model, train_loader, optimizer, scaler, device=device,
                             runtime=cfg["runtime"], accumulation=cfg["batch"]["accumulation"],
                             clipping=cfg["clipping"], scheduler=scheduler)
        validation = validate_epoch(forward_model, val_loader, device=device, runtime=cfg["runtime"])
        updates += train["optimizer_updates"]
        completed_epochs = epoch + 1
        record = {"epoch": completed_epochs, "train": train, "validation": validation,
                  "epoch_seconds": time.perf_counter() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        history.append(record)
        improved = is_better(validation, best, cfg["checkpoint"]["tolerance"], cfg["checkpoint"]["tie_break"])
        if improved:
            best = {"epoch": completed_epochs, "macro_f1_supported": validation["macro_f1_supported"],
                    "nll": validation["nll"]}
        if improved:
            save_checkpoint(folder / "checkpoint_best.pt", epoch=completed_epochs, updates=updates,
                            history=history, best=best, **checkpoint_args)
        save_checkpoint(folder / "checkpoint_last.pt", epoch=completed_epochs, updates=updates,
                        history=history, best=best, **checkpoint_args)
        write_history(folder, history)
        print(f"PILOT epoch {completed_epochs}/{cfg['budget']['max_epochs']}: val macro-F1={validation['macro_f1_supported']:.6f}", flush=True)
        if stop_requested():
            break
    write_history(folder, history)
    write_json(folder / "timing.json", [{"epoch": r["epoch"], "train": r["train"], "epoch_seconds": r["epoch_seconds"]} for r in history])
    write_json(folder / "memory.json", [{k: r[k] for k in ("epoch", "peak_allocated_bytes", "peak_reserved_bytes")} for r in history])
    return {"purpose": "pilot", "stage": "p1", "experiment": cfg["experiment"],
            "status": "complete" if completed_epochs == cfg["budget"]["max_epochs"] else "interrupted_at_epoch_boundary",
            "planned_epochs": cfg["budget"]["max_epochs"], "completed_epochs": completed_epochs,
            "optimizer_updates": updates, "planned_optimizer_updates": total_updates,
            "seed": cfg["seed"], "best_diagnostic_checkpoint": best, "identity": run_identity,
            "test_images_opened": 0, "official_experiment": False}


def main():
    p = parser(__doc__, resume=True)
    args = p.parse_args()
    cfg = load_config(args.config, data_root=args.data_root, weights=args.weights, runs_dir=args.runs_dir)
    if cfg["stage"] != "p1":
        p.error("Expected P1 config")
    if args.check_config:
        print("P1 config complete; no model/job run")
        return
    folder = create_run(cfg, args.run_id, args.resume)
    requested = {"stop": False}

    def request_stop(_signum, _frame):
        requested["stop"] = True

    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)
    with run_lock(folder):
        try:
            result = run(cfg, folder, args.resume, lambda: requested["stop"])
        except Exception as exc:
            write_json(folder / "run_summary.json", {"purpose": "pilot", "stage": "p1",
                       "experiment": cfg["experiment"], "status": "fail", "error_type": type(exc).__name__,
                       "error": str(exc), "resume_from_last_completed_epoch_only": True})
            raise
        write_json(folder / "run_summary.json", result)
    print(folder)


if __name__ == "__main__":
    main()
