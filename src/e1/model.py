"""Pinned offline ConvNeXt-Tiny and explicitly enumerated AdamW groups."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import convnext_tiny
from torchvision.models.convnext import CNBlock

from .config import load_recipe, sha256_file


class E1Model(nn.Module):
    def __init__(self, backbone: nn.Module, initialization: dict):
        super().__init__()
        seed = initialization.get("seed")
        if initialization.get("method") != "linear_default" or type(seed) is not int or seed not in {1001, 2001, 3001, 4001}:
            raise ValueError("Reviewed E1 initialization and official run seed required")
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.final_norm = backbone.classifier[0]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.classifier = nn.Linear(768, 124, bias=True)
        self.requires_grad_(True)
        self.initialization = {"method": "linear_default", "seed": seed, "fan_in": 768,
                               "weight": "kaiming_uniform_(a=sqrt(5))", "bias": "U(-1/sqrt(768),+1/sqrt(768))",
                               "rng": "isolated CPU; outer state restored", "torch_version": str(torch.__version__)}

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.final_norm(self.avgpool(self.features(images))).flatten(1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images))


def build_e1(weights_path: Path, expected_sha256: str, initialization: dict) -> tuple[nn.Module, dict]:
    pinned = load_recipe()["model"]["pretrained"]
    path = Path(weights_path)
    if expected_sha256 != pinned["sha256"]:
        raise ValueError("Only the full G1-pinned ImageNet digest is accepted")
    if path.name != pinned["filename"] or not path.is_file():
        raise ValueError("Exact offline pretrained filename required")
    if sha256_file(path) != expected_sha256:
        raise ValueError("Pretrained full SHA-256 mismatch")
    backbone = convnext_tiny(weights=None, stochastic_depth_prob=.1)
    backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model = E1Model(backbone, initialization)
    return model, {"pretrained": pinned, "initialization": model.initialization,
                   "max_stochastic_depth": .1, "depth_schedule": "TorchVision block-index linear 0..0.1",
                   "parameters": sum(p.numel() for p in model.parameters())}


def build_optimizer(model: nn.Module, config: dict) -> tuple[torch.optim.Optimizer, dict]:
    expected = load_recipe()["optimizer"]
    if {k: v for k, v in config.items() if k != "lr"} != expected or config.get("lr") not in load_recipe()["campaign"]["lr_candidates"]:
        raise ValueError("Reviewed AdamW configuration and LR required")
    groups = {"decay": [], "no_decay": []}
    names = {"decay": [], "no_decay": []}
    seen = set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError("Parameter appears in multiple optimizer groups")
            seen.add(id(parameter))
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name == "weight":
                category = "decay"
            elif name == "bias" or (isinstance(module, nn.LayerNorm) and name == "weight") or (isinstance(module, CNBlock) and name == "layer_scale"):
                category = "no_decay"
            else:
                raise ValueError(f"Unreviewed parameter category: {module_name}.{name}")
            groups[category].append(parameter)
            names[category].append(f"{module_name}.{name}".lstrip("."))
    if seen != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Trainable parameter omitted")
    optimizer = torch.optim.AdamW([{"params": groups["decay"], "weight_decay": .05},
                                   {"params": groups["no_decay"], "weight_decay": 0.}],
                                  lr=config["lr"], betas=(.9, .999), eps=1e-8)
    return optimizer, names
