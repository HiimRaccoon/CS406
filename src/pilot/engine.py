"""Minimal bounded-pilot mechanics; no search, early budget extension, or test evaluation."""
from __future__ import annotations

import contextlib
import itertools
import math
import time

import torch
from torch.nn import functional as F

from .metrics import ClassificationMetrics


class NumericalFailure(RuntimeError):
    pass


class WarmupCosine:
    def __init__(self, optimizer, total_updates, warmup_updates, start_factor, min_lr):
        if not 0 <= warmup_updates < total_updates:
            raise ValueError("Warmup must be shorter than planned optimizer-update budget")
        self.optimizer, self.total, self.warmup = optimizer, total_updates, warmup_updates
        self.start_factor, self.min_lr = start_factor, min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.completed = 0
        self._apply()

    def _apply(self):
        u = min(self.completed, self.total - 1)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            if u < self.warmup:
                # First update uses start_factor; cosine begins at base LR.
                lr = base * (self.start_factor + (1 - self.start_factor) * u / self.warmup)
            else:
                remaining = self.total - self.warmup
                progress = (u - self.warmup) / (remaining - 1) if remaining > 1 else 1.
                lr = self.min_lr + .5 * (base - self.min_lr) * (1 + math.cos(math.pi * progress))
            group["lr"] = lr

    def step(self):
        self.completed += 1
        self._apply()

    def state_dict(self):
        return {k: getattr(self, k) for k in ("total", "warmup", "start_factor", "min_lr", "base_lrs", "completed")}

    def load_state_dict(self, state):
        current = self.state_dict()
        if any(state[k] != current[k] for k in current if k != "completed"):
            raise ValueError("Cannot change scheduler horizon on resume")
        self.completed = state["completed"]
        self._apply()


def autocast_context(runtime, device):
    if not runtime["amp"]["enabled"]:
        return contextlib.nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[runtime["amp"]["dtype"]]
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scaler(runtime):
    policy = runtime["scaler"]
    kwargs = {k: policy[k] for k in ("init_scale", "growth_factor", "backoff_factor", "growth_interval")
              if policy.get("enabled")}
    return torch.amp.GradScaler("cuda", enabled=policy["enabled"], **kwargs)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def move_views(views, device):
    return {key: value.to(device, non_blocking=True) for key, value in views.items()}


def train_window(model, loader, optimizer, scaler, *, device, runtime, accumulation,
                 clipping, scheduler=None, max_updates=None, iterator=None):
    """A full epoch or finite update window; exact sample weighting for short tail groups."""
    model.train()
    updates = samples = microbatches = 0
    loss_total = wait_total = 0.
    lrs = []
    synchronize(device)
    started = time.perf_counter()
    if iterator is None:
        iterator = iter(loader)
        wait_total += time.perf_counter() - started
    while max_updates is None or updates < max_updates:
        wait_start = time.perf_counter()
        group = list(itertools.islice(iterator, accumulation))
        wait_total += time.perf_counter() - wait_start
        if not group:
            break
        denominator = sum(len(targets) for _, targets in group)
        optimizer.zero_grad(set_to_none=True)
        for views, targets in group:
            views, targets = move_views(views, device), targets.to(device, non_blocking=True)
            with autocast_context(runtime, device):
                logits = model(views)
                loss_sum = F.cross_entropy(logits, targets, reduction="sum")
                loss = loss_sum / denominator
            if not torch.isfinite(loss):
                raise NumericalFailure("Nonfinite training loss")
            scaler.scale(loss).backward()
            loss_total += loss_sum.detach().float().item()
            samples += len(targets)
            microbatches += 1
        scaler.unscale_(optimizer)
        finite_grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if not finite_grads or not torch.stack(finite_grads).all():
            raise NumericalFailure("Nonfinite unscaled gradients; no silent skipped update or budget extension")
        if clipping["enabled"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping["max_norm"], error_if_nonfinite=True)
        lrs.append(optimizer.param_groups[0]["lr"])
        scaler.step(optimizer)
        scaler.update()
        updates += 1
        if scheduler is not None:
            scheduler.step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    if samples == 0:
        raise ValueError("Empty training window")
    return {"loss": loss_total / samples, "samples": samples, "optimizer_updates": updates,
            "microbatches": microbatches, "seconds": elapsed, "images_per_second": samples / elapsed,
            "mean_microbatch_seconds": elapsed / microbatches, "mean_update_seconds": elapsed / updates,
            "data_wait_seconds": wait_total, "data_wait_fraction": wait_total / elapsed,
            "lr_first": lrs[0], "lr_last": lrs[-1]}


@torch.no_grad()
def validate_epoch(model, loader, *, device, runtime):
    model.eval()
    metric = ClassificationMetrics()
    for views, targets in loader:
        targets = targets.to(device, non_blocking=True)
        with autocast_context(runtime, device):
            logits = model(move_views(views, device))
            loss_sum = F.cross_entropy(logits, targets, reduction="sum")
        if not torch.isfinite(loss_sum):
            raise NumericalFailure("Nonfinite validation loss")
        metric.update(logits, targets, loss_sum.item())
    return metric.compute()


def is_better(metrics, best, tolerance, tie_break):
    if best is None:
        return True
    delta = metrics["macro_f1_supported"] - best["macro_f1_supported"]
    return delta > tolerance or (abs(delta) <= tolerance and tie_break == "nll" and metrics["nll"] < best["nll"])
