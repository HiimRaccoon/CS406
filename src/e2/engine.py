"""E2-owned dictionary-view epochs; original-image denominators and dual transfer."""
from __future__ import annotations
import itertools
import time
import torch
from torch.nn import functional as F
from ..e1.engine import UpdateScheduler, select_checkpoint, NumericalFailure
from .metrics import Metrics124, rank_logits
from .data import move_views


def _batch(targets, paths):
    if (not isinstance(targets,torch.Tensor) or targets.ndim != 1 or targets.dtype != torch.int64
            or len(paths) != len(targets) or not 1 <= len(targets) <= 2
            or (targets < 0).any() or (targets >= 124).any()):
        raise ValueError('E2 requires one label/path per original image')


def _fp32(runtime, scaler=None):
    if (runtime.get("dtype") != "float32" or runtime.get("amp", {}).get("enabled", False)
            or scaler is not None):
        raise ValueError("Reviewed E2 mechanics require FP32 with scaler OFF")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, scheduler, scaler, *, device, runtime: dict, accumulation: int, clipping: dict) -> dict:
    _fp32(runtime, scaler)
    if type(accumulation) is not int or accumulation != 1 or clipping["enabled"]:
        raise ValueError("Positive accumulation and reviewed clipping OFF required")
    model.train()
    _sync(device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    iterator = iter(loader)
    samples = updates = microbatches = 0
    loss_total = wait_total = 0.
    first_lr = last_lr = None
    while True:
        wait_start = time.perf_counter()
        group = list(itertools.islice(iterator, accumulation))
        wait_total += time.perf_counter() - wait_start
        if not group:
            break
        if scheduler is not None and scheduler.completed >= scheduler.total_updates:
            raise ValueError("No optimizer updates beyond original horizon")
        denominator = sum(len(y) for _, y, _ in group)
        optimizer.zero_grad(set_to_none=True)
        for images, targets, paths in group:
            _batch(targets, paths)
            images = move_views(images, device=device, batch_size=len(targets))
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            if tuple(logits.shape) != (len(targets),124):
                raise ValueError('E2 logits must be [original_images,124]')
            loss_sum = F.cross_entropy(logits, targets, reduction="sum", label_smoothing=0.)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss_sum):
                raise NumericalFailure("Nonfinite training logits/loss")
            (loss_sum / denominator).backward()
            loss_total += loss_sum.detach().item()
            samples += len(targets)
            microbatches += 1
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise NumericalFailure("Nonfinite or missing gradients; no silent skipped update")
        last_lr = optimizer.param_groups[0]["lr"]
        first_lr = last_lr if first_lr is None else first_lr
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        updates += 1
    _sync(device)
    seconds = time.perf_counter() - started
    if samples == 0:
        raise ValueError("Empty training epoch")
    return {"loss": loss_total / samples, "samples": samples, "optimizer_updates": updates, "microbatches": microbatches,
            "lr_first": first_lr, "lr_last": last_lr, "seconds": seconds, "images_per_second": samples / seconds,
            "data_wait_seconds": wait_total, "data_wait_fraction": wait_total / seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}


@torch.no_grad()
def evaluate(model, loader, *, device, runtime: dict, n_train: dict, labels: list[str], collect_predictions: bool = False) -> dict:
    _fp32(runtime)
    model.eval()
    metric, predictions, seen = Metrics124(), [], set()
    for images, targets, paths in loader:
        if len(paths) != len(targets) or any(p in seen for p in paths) or len(set(paths)) != len(paths):
            raise ValueError("Duplicate/missing evaluation paths")
        seen.update(paths)
        _batch(targets, paths)
        images = move_views(images, device=device, batch_size=len(targets))
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        if not torch.isfinite(loss_sum):
            raise NumericalFailure("Nonfinite validation loss")
        metric.update(logits, targets, loss_sum.item())
        if collect_predictions:
            ranking = rank_logits(logits).cpu()
            for path, target, ranked in zip(paths, targets.cpu().tolist(), ranking.tolist()):
                predictions.append({"relative_path": path, "label_id": target, "predicted_label_id": ranked[0], "top5": ranked[:5]})
    result = metric.compute(n_train, labels)
    if collect_predictions:
        result["predictions"] = predictions
    return result
