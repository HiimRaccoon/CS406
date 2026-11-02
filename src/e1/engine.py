"""FP32 epoch mechanics, update-based schedule, and chronological selection."""
from __future__ import annotations

import itertools
import math
import time

import torch
from torch.nn import functional as F

from .metrics import Metrics124, rank_logits


class NumericalFailure(RuntimeError):
    pass


class UpdateScheduler:
    def __init__(self, optimizer, schedule: dict, total_updates: int):
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.warmup = schedule["warmup_updates"]
        self.start = schedule["warmup_start_factor"]
        self.floor = schedule["cosine_floor_lr"]
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        if (type(total_updates) is not int or total_updates < 2 or type(self.warmup) is not int
                or not 0 <= self.warmup < total_updates - 1 or not 0 < self.start <= 1
                or not math.isfinite(self.floor) or not 0 <= self.floor <= min(self.base_lrs)):
            raise ValueError("Invalid complete update schedule")
        self.completed = 0
        self._apply()

    def _apply(self):
        u = min(self.completed, self.total_updates - 1)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            if u == self.total_updates - 1:
                lr = self.floor
            elif u < self.warmup:
                lr = base * (self.start + (1 - self.start) * u / self.warmup)
            else:
                lr = self.floor + (base - self.floor) / 2 * (1 + math.cos(math.pi * (u - self.warmup) / (self.total_updates - self.warmup - 1)))
            g["lr"] = lr

    def step(self):
        if self.completed >= self.total_updates:
            raise ValueError("Original scheduler budget exhausted")
        self.completed += 1
        self._apply()

    def state_dict(self):
        return {k: getattr(self, k) for k in ("total_updates", "warmup", "start", "floor", "base_lrs", "completed")}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (set(state) != set(expected) or any(state[k] != expected[k] for k in expected if k != "completed")
                or type(state["completed"]) is not int or not 0 <= state["completed"] <= self.total_updates):
            raise ValueError("Changed scheduler configuration/horizon or invalid progress")
        self.completed = state["completed"]
        self._apply()


def select_checkpoint(current: dict, selected: dict | None) -> bool:
    value = current["macro_f1_supported"]
    if not math.isfinite(value) or not 0 <= value <= 1 or type(current["epoch"]) is not int or not 1 <= current["epoch"] <= 8:
        raise ValueError("Invalid completed checkpoint metric")
    if selected is None:
        return True
    if current["epoch"] <= selected["epoch"]:
        raise ValueError("Checkpoint comparisons must be chronological")
    return value - selected["macro_f1_supported"] > 1e-6


def _fp32(runtime, scaler=None):
    if (runtime.get("dtype") != "float32" or runtime.get("amp", {}).get("enabled", False)
            or (scaler is not None and scaler.is_enabled())):
        raise ValueError("Reviewed E1 mechanics require FP32 with scaler OFF")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, scheduler, scaler, *, device, runtime: dict, accumulation: int, clipping: dict) -> dict:
    _fp32(runtime, scaler)
    if type(accumulation) is not int or accumulation < 1 or clipping["enabled"]:
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
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            logits = model(images)
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
        targets = targets.to(device, non_blocking=True)
        logits = model(images.to(device, non_blocking=True))
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
