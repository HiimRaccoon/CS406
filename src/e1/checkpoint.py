"""Atomic official boundaries; best is reporting-only, last is resume-only."""
from __future__ import annotations

import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint


def _atomic(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    def write(p):
        with p.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    _atomic(path, write)


def atomic_text(path: Path, text: str) -> None:
    def write(p):
        with p.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    _atomic(path, write)


def _rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _progress(progress, config, scheduler_state):
    budget = config["recipe"]["budget"]
    per_epoch = config["recipe"]["batch"]["updates_per_epoch"]
    e, u = progress["completed_epoch"], progress["updates"]
    if (budget["max_epochs"] != 8 or type(e) is not int or not 0 <= e <= 8 or type(u) is not int
            or u != e * per_epoch or len(progress["history"]) != e
            or [r["epoch"] for r in progress["history"]] != list(range(1, e + 1))
            or scheduler_state["total_updates"] != budget["planned_max_updates"]
            or scheduler_state["completed"] != u):
        raise ValueError("Invalid original-budget epoch boundary")
    selected = progress["selected"]
    if e == 0:
        if selected is not None or progress["selected_weights"] is not None:
            raise ValueError("Invalid boundary-zero selection")
    elif (not isinstance(selected, dict) or type(selected.get("epoch")) is not int
          or not 1 <= selected["epoch"] <= e or not math.isfinite(selected["macro_f1_supported"])
          or not 0 <= selected["macro_f1_supported"] <= 1 or not progress["selected_weights"]):
        raise ValueError("Missing or invalid selected-checkpoint state")


def save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, config: dict, progress: dict, identity: dict, loaders: dict) -> None:
    path = Path(path)
    if path.name not in {"best.pt", "last.pt"}:
        raise ValueError("Official checkpoint role must be best.pt or last.pt")
    sched = scheduler.state_dict()
    _progress(progress, config, sched)
    if path.name == "best.pt" and progress["selected"] is None:
        raise ValueError("No selected checkpoint at boundary zero")
    state = {"kind": "official_e1_checkpoint_v1", "role": path.stem,
             "model": progress["selected_weights"] if path.stem == "best" else model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": sched,
             "scaler": scaler.state_dict() if scaler is not None else None,
             "config": config, "config_sha256": fingerprint(config), "identity": identity,
             "progress": progress, "rng": _rng(),
             "loader_rng": {k: loader.generator.get_state() for k, loader in loaders.items()}}
    _atomic(path, lambda p: torch.save(state, p))


def read_checkpoint(path: Path, *, config: dict, identity: dict) -> dict:
    # Only trusted local official artifacts are deserialized; pretrained loading uses weights_only=True.
    state = torch.load(path, map_location="cpu", weights_only=False)
    required = {"kind", "role", "model", "optimizer", "scheduler", "scaler", "config", "config_sha256", "identity", "progress", "rng", "loader_rng"}
    try:
        if (set(state) != required or state["kind"] != "official_e1_checkpoint_v1" or state["role"] not in {"best", "last"}
                or state["config_sha256"] != fingerprint(config) or fingerprint(state["config"]) != fingerprint(config)
                or state["identity"] != identity):
            raise ValueError("Checkpoint identity/config/kind mismatch")
        _progress(state["progress"], config, state["scheduler"])
        if state["role"] == "best" and state["progress"]["selected"] is None:
            raise ValueError("Unselected best checkpoint")
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete official checkpoint") from exc
    return state


def restore_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, config: dict, identity: dict, loaders: dict) -> dict:
    if Path(path).name != "last.pt":
        raise ValueError("Resume only from last.pt, never best.pt")
    state = read_checkpoint(path, config=config, identity=identity)
    if state["role"] != "last" or set(state["loader_rng"]) != set(loaders):
        raise ValueError("Wrong resume role or loader state")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if (state["scaler"] is None) != (scaler is None):
        raise ValueError("Scaler policy changed on resume")
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    for key, loader in loaders.items():
        loader.generator.set_state(state["loader_rng"][key])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["cpu"])
    if state["rng"]["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Cannot restore CUDA run on CPU")
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return state["progress"]
