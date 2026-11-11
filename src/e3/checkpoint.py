"""Atomic authoritative last state, safe tensor loading and deterministic replay."""
from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import torch

from .config import DIMENSIONS, fingerprint, payload, read_json, sha256_file, validate_record
from .engine import select_head
from .io import atomic_json, atomic_write
from .model import tensor_state_sha256


def _rng_state():
    numpy = np.random.get_state()
    return {"python":random.getstate(),"numpy":{"algorithm":numpy[0],"keys":torch.from_numpy(numpy[1].astype(np.int64)),
            "position":numpy[2],"gauss":numpy[3],"cached":numpy[4]},"cpu":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def capture_head_state(*, head, optimizer, scheduler, loaders, sampler, progress: dict) -> dict:
    return copy.deepcopy({"head":{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                          "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                          "loader_rng":{k:v.generator.get_state() for k,v in loaders.items()},
                          "sampler":sampler.state_dict(),"rng":_rng_state(),"progress":progress})


def restore_head_state(state: dict, *, head, optimizer, scheduler, loaders, sampler) -> dict:
    if set(state["loader_rng"]) != set(loaders):
        raise ValueError("Loader identities changed on resume")
    head.load_state_dict(state["head"],strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampler"])
    for name,loader in loaders.items():
        loader.generator.set_state(state["loader_rng"][name])
    rng=state["rng"]; random.setstate(rng["python"])
    n=rng["numpy"]; np.random.set_state((n["algorithm"],n["keys"].numpy().astype(np.uint32),n["position"],n["gauss"],n["cached"]))
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Cannot restore CUDA RNG on a different execution device")
        torch.cuda.set_rng_state_all(rng["cuda"])
    return copy.deepcopy(state["progress"])


def _validate_progress(progress, budget, scheduler):
    try:
        window=progress["window"]
        if (type(window) is not int or not 0 <= window <= budget["windows"]
                or progress["presentations"] != window*budget["presentations_per_window"]
                or progress["updates"] != window*budget["updates_per_window"]
                or scheduler["completed"] != progress["updates"] or len(progress["history"]) != window):
            raise ValueError("Invalid complete-window boundary or original budget")
        selected=None
        for i,row in enumerate(progress["history"],1):
            if (row["window"] != i or row["updates"] != i*budget["updates_per_window"]
                    or row["presentations"] != i*budget["presentations_per_window"]
                    or row["train"]["presentations"] != budget["presentations_per_window"]
                    or row["train"]["updates"] != budget["updates_per_window"]):
                raise ValueError("History accounting is incomplete or reordered")
            candidate={"window":i,"updates":row["updates"],"macro_f1_supported":row["validation"]["macro_f1_supported"]}
            if select_head(candidate,selected): selected=candidate
        if selected != progress["selected"] or (progress["selected_weights"] is None) != (window==0):
            raise ValueError("Selected point does not follow complete chronological history")
        fingerprint(progress["history"])
    except (KeyError,TypeError) as exc:
        raise ValueError("Incomplete committed progress") from exc


def _validate_head_tensors(state,config):
    dimension=DIMENSIONS[config["family"]]
    if (set(state)!={"weight","bias"} or tuple(state["weight"].shape)!=(124,dimension)
            or tuple(state["bias"].shape)!=(124,)
            or any(v.dtype!=torch.float32 for v in state.values())):
        raise ValueError("Classifier state shape/dtype differs from declared family")
    tensor_state_sha256(state)


def publish_derived(path: Path, envelope: dict) -> dict:
    root=Path(path).parent; config=envelope["config"]; progress=envelope["progress"]
    _validate_progress(progress,config["original_budget"],envelope["scheduler"])
    declaration=root/"declaration.json"
    if declaration.exists():
        if read_json(declaration) != config: raise ValueError("Declaration bytes/identity changed")
    else:
        atomic_json(declaration,config)
    atomic_json(root/"history.json",progress["history"])
    selected={"status":"NOT_SELECTED","reason":"boundary_zero_no_validation"}
    if progress["selected"] is not None:
        weights=progress["selected_weights"]
        head_value={"kind":"e3_selected_head_tensor_v1","run_id":config["run_id"],"state":weights,
                    "tensor_state_sha256":tensor_state_sha256(weights),"selected":progress["selected"]}
        atomic_write(root/"best.pt",lambda stream:torch.save(head_value,stream))
        selected={"status":"SELECTED","selected_window":progress["selected"]["window"],
                  "selected_update":progress["selected"]["updates"],
                  "macro_f1_supported":progress["selected"]["macro_f1_supported"],"head_payload":payload(root/"best.pt",root)}
    header={"kind":"e3_head_checkpoint_v1","run_id":config["run_id"],"stage":config["stage"],
            "declaration_sha256":sha256_file(declaration),
            **{k:config[k] for k in ("source","cache","initialization","profile","runtime")},
            "committed_window":progress["window"],"committed_presentations":progress["presentations"],
            "committed_updates":progress["updates"],"full_state_payload":payload(path,root),
            "history_payload":payload(root/"history.json",root),"selected_head_state":selected}
    validate_record(header,"head_checkpoint")
    atomic_json(root/"last.json",header)
    return header


def save_head_checkpoint(path: Path, *, state: dict, config: dict, identity: dict, role: str) -> None:
    path=Path(path)
    if role != "last" or path.name != "last.pt":
        raise ValueError("Only authoritative last.pt commits training state; best is derived")
    validate_record(config)
    _validate_progress(state["progress"],config["original_budget"],state["scheduler"])
    _validate_head_tensors(state["head"],config)
    if state["progress"]["selected_weights"] is not None:
        _validate_head_tensors(state["progress"]["selected_weights"],config)
    envelope={"kind":"e3_full_head_state_v1","role":"last","config":config,
              "config_sha256":fingerprint(config),"identity":identity,**copy.deepcopy(state)}
    atomic_write(path,lambda stream:torch.save(envelope,stream))
    publish_derived(path,envelope)


def read_head_checkpoint(path: Path, *, config: dict, identity: dict) -> dict:
    path=Path(path)
    if path.name != "last.pt":
        raise ValueError("Resume only from authoritative same-run last.pt")
    # Safe tensors/primitives permit recovery when derived JSON publication failed after
    # the atomic full-state commit. Never deserialize Python objects from source runs here.
    state=torch.load(path,map_location="cpu",weights_only=True)
    try:
        if (state["kind"] != "e3_full_head_state_v1" or state["role"] != "last"
                or state["config_sha256"] != fingerprint(config) or state["config"] != config
                or state["identity"] != identity):
            raise ValueError("Foreign checkpoint kind/config/source/cache/init/profile/identity")
        validate_record(config)
        _validate_progress(state["progress"],config["original_budget"],state["scheduler"])
        _validate_head_tensors(state["head"],config)
        if state["progress"]["selected_weights"] is not None:
            _validate_head_tensors(state["progress"]["selected_weights"],config)
    except (KeyError,TypeError) as exc:
        raise ValueError("Incomplete E3 head checkpoint") from exc
    return state
