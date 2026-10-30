"""One shared representation; no official experiment or E3 builders."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from ..preprocessing.geoguessr_v1 import sha256_file

WEIGHTS = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
WEIGHT_FILENAME = "convnext_tiny-983f1562.pth"


def initialize_classifier(head: nn.Linear, config: dict) -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        if config["method"] == "linear_default":
            head.reset_parameters()
            rule = "torch.nn.Linear.reset_parameters: kaiming_uniform_(a=sqrt(5)); bias U(-1/sqrt(fan_in),1/sqrt(fan_in))"
        elif config["method"] == "trunc_normal":
            nn.init.trunc_normal_(head.weight, mean=0., std=config["std"], a=-2., b=2.)
            nn.init.constant_(head.bias, config["bias"])
            rule = "torch.nn.init.trunc_normal_(mean=0,std=config.std,a=-2,b=2); constant bias=config.bias"
        else:
            raise ValueError("Initialization must be explicitly selected")
    return {**config, "rule": rule, "torch_version": torch.__version__, "fan_in": head.in_features,
            "rng_behavior": "isolated CPU RNG seeded immediately before initialization; outer RNG restored"}


class PilotModel(nn.Module):
    def __init__(self, backbone, experiment, initialization):
        super().__init__()
        if experiment not in {"e1", "e1hr", "e2"}:
            raise ValueError("Unsupported pilot profile")
        self.experiment = experiment
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.final_norm = backbone.classifier[0]
        self.classifier = nn.Linear(1536 if experiment == "e2" else 768, 124)
        self.initialization = initialize_classifier(self.classifier, initialization)

    def encode(self, image):
        return self.final_norm(self.avgpool(self.features(image))).flatten(1)

    def forward(self, views):
        global_feature = self.encode(views["global"])
        if self.experiment == "e2":
            local = views["local"]
            if local.ndim != 5 or local.shape[1] != 4:
                raise ValueError("E2 requires [batch,4,3,height,width] local views")
            features = self.encode(local.flatten(0, 1)).reshape(local.shape[0], 4, 768).mean(1)
            global_feature = torch.cat((global_feature, features), dim=1)
        return self.classifier(global_feature)


def build_model(experiment, weights_path, initialization, expected_sha256=None):
    path = Path(weights_path)
    if path.name != WEIGHT_FILENAME or not path.is_file():
        raise FileNotFoundError(f"Offline weight file required: {WEIGHT_FILENAME}; set CS406_WEIGHTS_PATH")
    digest = sha256_file(path)
    if not digest.startswith("983f1562") or (expected_sha256 and digest != expected_sha256):
        raise ValueError("Pretrained weight SHA-256 mismatch")
    # Architecture is built with the enum's exact state dictionary, offline. Never invoke transforms/download.
    backbone = convnext_tiny(weights=None, stochastic_depth_prob=.1)
    backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model = PilotModel(backbone, experiment, initialization)
    return model, {"enum": str(WEIGHTS), "filename": path.name, "sha256": digest,
                   "initialization": model.initialization, "max_drop_path_rate": .1,
                   "depth_schedule": "TorchVision block-index linear ramp 0..0.1"}


def build_optimizer(model, cfg):
    decay, no_decay, names, seen = [], [], {"decay": [], "no_decay": []}, set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError("Parameter grouped twice")
            seen.add(id(parameter))
            key = f"{module_name}.{name}".lstrip(".")
            if name == "weight" and isinstance(module, (nn.Conv2d, nn.Linear)):
                decay.append(parameter)
                names["decay"].append(key)
            else:  # bias, LayerNorm, and ConvNeXt layer_scale: decay only Conv/Linear weights
                no_decay.append(parameter)
                names["no_decay"].append(key)
    if len(seen) != sum(p.requires_grad for p in model.parameters()):
        raise ValueError("Trainable parameter omitted from optimizer")
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .05},
                                  {"params": no_decay, "weight_decay": 0.}],
                                 lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"])
    return optimizer, names
