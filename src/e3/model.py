"""Fresh paired linear heads and exactly two new optimizer tensors."""
from __future__ import annotations

import copy
import hashlib
import json

import torch
from torch import nn

from .config import DIMENSIONS, SEEDS, fingerprint, policy, validate_common_protocol
from ..e1.data import derived_seed


def tensor_state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        tensor = tensor.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite head state")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], ensure_ascii=True).encode()
        digest.update(len(header).to_bytes(8,"little")); digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def head_state_sha256(head) -> str:
    return tensor_state_sha256(head.state_dict())


def make_paired_heads(*, family: str, seed: int, protocol: dict, stage: str, root_seed: int):
    validate_common_protocol(protocol)
    if family not in DIMENSIONS or seed not in SEEDS or stage not in {"pilot", "final"}:
        raise ValueError("Invalid paired head family/seed/stage")
    if (stage == "pilot" and (seed != 2001 or root_seed != 1001)) or (stage == "final" and root_seed != seed):
        raise ValueError("Wrong source/root seed role")
    actual_seed = derived_seed(root_seed,"E3-v1",stage,family,seed,"head-init")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(actual_seed)
        first = nn.Linear(DIMENSIONS[family],124,bias=True,device="cpu",dtype=torch.float32)
    second = copy.deepcopy(first)
    identity = {"tensor_state_sha256": head_state_sha256(first), "initialization_seed": actual_seed,
                "initializer_policy_sha256": fingerprint(policy()["initializer"]), "fan_in": DIMENSIONS[family]}
    return first, second, identity


def build_head_optimizer(head, *, protocol: dict):
    validate_common_protocol(protocol)
    if (not isinstance(head, nn.Linear) or head.in_features not in DIMENSIONS.values() or head.out_features != 124
            or head.bias is None or list(dict(head.named_parameters())) != ["weight","bias"]
            or any(not p.requires_grad or p.dtype != torch.float32 for p in head.parameters())):
        raise ValueError("Optimizer requires exactly new biased FP32 E3 head tensors")
    config = protocol["optimizer"]
    optimizer = torch.optim.AdamW(
        [{"params":[head.weight],"weight_decay":config["weight_decay_groups"]["weight"]},
         {"params":[head.bias],"weight_decay":config["weight_decay_groups"]["bias"]}],
        lr=protocol["base_lr"], betas=tuple(config["betas"]), eps=config["eps"],
        **{k:config[k] for k in ("amsgrad","maximize","foreach","fused","capturable","differentiable")})
    return optimizer,{"weight":["weight"],"bias":["bias"]}
