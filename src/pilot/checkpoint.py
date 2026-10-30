"""Atomic, epoch-boundary checkpoints for bounded pilots only."""
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import fingerprint


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(value, path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, *, model, optimizer, scheduler, scaler, epoch, updates, cfg,
                    history, best, loader_generators, identity):
    atomic_save({"kind": "bounded_pilot_checkpoint", "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "global_optimizer_update": updates,
                 "config": cfg, "config_sha256": fingerprint(cfg), "training_seed": cfg["seed"],
                 "rng": rng_state(), "loader_rng": {k: g.get_state() for k, g in loader_generators.items()},
                 "best": best, "history": history, "identity": identity}, path)


def load_checkpoint(path, *, model, optimizer, scheduler, scaler, cfg, loader_generators, identity):
    # Checkpoints are trusted local artifacts, never downloaded untrusted pickle inputs.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("kind") != "bounded_pilot_checkpoint" or state["config_sha256"] != fingerprint(cfg):
        raise ValueError("Resume config/budget/seed differs from the planned pilot")
    if state["identity"] != identity:
        raise ValueError("Resume dataset/preprocessing/weights/environment identity differs")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    for key, generator in loader_generators.items():
        generator.set_state(state["loader_rng"][key])
    restore_rng(state["rng"])
    return state
