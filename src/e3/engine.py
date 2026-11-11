"""Standalone cached-head updates, exact presentation accounting and full-split metrics."""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import DIMENSIONS, fingerprint, validate_common_protocol
from .metrics import Metrics124, rank_logits


class HeadSchedule:
    def __init__(self, optimizer, schedule, base_lr):
        self.optimizer, self.schedule, self.base_lr = optimizer, dict(schedule), base_lr
        self.completed = 0

    def set_lr(self):
        u = self.completed
        total, warmup = self.schedule["total_updates"], self.schedule["warmup_updates"]
        if not 0 <= u < total:
            raise ValueError("Scheduler exhausted; budget cannot extend")
        floor = self.schedule["cosine_floor_factor"]
        factor = (u+1)/warmup if u < warmup else floor + (1-floor)*(1+math.cos(math.pi*(u-warmup)/(total-warmup-1)))/2
        lr = self.base_lr * factor
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self):
        if self.completed >= self.schedule["total_updates"]:
            raise ValueError("Extra optimizer update forbidden")
        self.completed += 1

    def state_dict(self):
        return {"completed": self.completed, "schedule_sha256": fingerprint(self.schedule), "base_lr": self.base_lr}

    def load_state_dict(self, state):
        if (set(state) != {"completed","schedule_sha256","base_lr"}
                or state["schedule_sha256"] != fingerprint(self.schedule) or state["base_lr"] != self.base_lr
                or type(state["completed"]) is not int or not 0 <= state["completed"] <= self.schedule["total_updates"]):
            raise ValueError("Scheduler identity/cursor mismatch")
        self.completed = state["completed"]


def make_head_scheduler(optimizer, *, protocol: dict):
    validate_common_protocol(protocol)
    return HeadSchedule(optimizer,protocol["schedule"],protocol["base_lr"])


def _train_batches(head, loader, optimizer, scheduler, *, accumulation, clipping, device):
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("Positive accumulation required")
    head.train()
    iterator = iter(loader)
    presentations, updates, loss_total = 0, 0, 0.
    while group := list(itertools.islice(iterator,accumulation)):
        count = sum(len(batch[1]) for batch in group)
        if not count:
            raise ValueError("Empty accumulation group")
        scheduler.set_lr()
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.
        for batch in group:
            features, targets = batch[:2]
            features, targets = features.to(device), targets.to(device)
            logits = head(features)
            if logits.shape != (len(targets),124) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite [B,124] head logits")
            loss = F.cross_entropy(logits,targets,reduction="sum",label_smoothing=0.)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite head CE")
            (loss/count).backward()
            group_loss += float(loss.detach())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise ValueError("Missing/nonfinite classifier gradient")
        torch.nn.utils.clip_grad_norm_(head.parameters(), **clipping)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in head.parameters()):
            raise ValueError("Nonfinite classifier state; restore committed boundary")
        scheduler.step()
        presentations += count; updates += 1; loss_total += group_loss
    if not presentations:
        raise ValueError("Empty training window")
    return {"presentations":presentations,"updates":updates,"ce":loss_total/presentations}


def train_window(head, loader, optimizer, scheduler, *, protocol: dict, device) -> dict:
    validate_common_protocol(protocol)
    if not isinstance(head,nn.Linear) or head.in_features not in DIMENSIONS.values() or head.out_features != 124:
        raise ValueError("Only the standalone E3 linear head trains")
    budget, clip = protocol["budget"], protocol["optimizer"]["clipping"]
    result = _train_batches(head,loader,optimizer,scheduler,accumulation=budget["accumulation"],
                            clipping={k:clip[k] for k in ("max_norm","norm_type","error_if_nonfinite","foreach")},device=device)
    if result["presentations"] != budget["presentations_per_window"] or result["updates"] != budget["updates_per_window"]:
        raise ValueError("Incomplete or excessive presentation/update window; no commit")
    return result


@torch.no_grad()
def _evaluate_batches(head, loader, *, records, n_train, labels, device, collect_predictions):
    head.eval()
    meter, predictions, cursor = Metrics124(), [], 0
    for features, targets, keys in loader:
        expected = records[cursor:cursor+len(targets)]
        if (list(keys) != [r["relative_path"] for r in expected]
                or targets.tolist() != [r["label_id"] for r in expected]):
            raise ValueError("Validation keys/labels/coverage/order differ")
        logits = head(features.to(device)); target = targets.to(device)
        losses = F.cross_entropy(logits,target,reduction="none",label_smoothing=0.)
        meter.update(logits,target,float(losses.sum()))
        if collect_predictions:
            ranks = rank_logits(logits)[:,:5].cpu().tolist()
            predictions.extend({"relative_path":key,"label_id":int(label),"prediction":rank[0],"top5":rank,"nll":float(loss)}
                               for key,label,rank,loss in zip(keys,targets.tolist(),ranks,losses.cpu().tolist()))
        cursor += len(targets)
    if cursor != len(records):
        raise ValueError("Truncated validation split")
    return {"metrics":meter.compute(n_train,labels),"predictions":predictions}


def evaluate_head(head, loader, *, manifest, device, collect_predictions: bool) -> dict:
    records = manifest.records("val")
    if len(records) != 7497:
        raise ValueError("Evaluation requires complete fixed validation membership")
    return _evaluate_batches(head,loader,records=records,n_train=manifest.n_train,labels=manifest.labels,
                             device=device,collect_predictions=collect_predictions)


def select_head(current: dict, selected: dict | None) -> bool:
    metric = current["macro_f1_supported"]
    if not math.isfinite(metric) or not 0 <= metric <= 1:
        raise ValueError("Invalid supported Macro-F1")
    return selected is None or metric-selected["macro_f1_supported"] > 1e-6
