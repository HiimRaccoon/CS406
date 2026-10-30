"""Exact v1 geometry; tensor float32 bilinear antialias, never weights.transforms."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_SHA256 = "4f49dd41738168cfc65d4322fbe1b659ef5595229bb487ad86dd671a26a8903f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_preprocessing(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs/preprocessing/geoguessr_v1.json"
    if not path.is_absolute():
        path = REPO_ROOT / path
    if sha256_file(path) != PREPROCESS_SHA256:
        raise ValueError("Frozen preprocessing SHA-256 mismatch; do not rewrite v1")
    sidecar = path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != PREPROCESS_SHA256:
        raise ValueError("Preprocessing checksum sidecar mismatch")
    return json.loads(path.read_text(encoding="utf-8"))



def half_up(value: float) -> int:
    return math.floor(value + 0.5)


@dataclass(frozen=True)
class Photometric:
    brightness: float
    contrast: float


class Preprocessor:
    def __init__(self, experiment="e1", global_wh=None, config_path=None):
        if experiment not in {"e1", "e1hr", "e2"}:
            raise ValueError("Only E1/E1-HR/E2 pilot views are supported")
        self.config = frozen_preprocessing(config_path)
        self.experiment = experiment
        self.global_wh = tuple(global_wh or (896, 384))
        if len(self.global_wh) != 2 or any(type(v) is not int or v < 32 for v in self.global_wh):
            raise ValueError("global_wh must be two integer dimensions >=32")
        if experiment != "e1hr" and self.global_wh != (896, 384):
            raise ValueError("E1/E2 global geometry is frozen at 896x384")
        self.mean = torch.tensor(self.config["normalization"]["mean"], dtype=torch.float32)[:, None, None]
        self.std = self.config["normalization"]["std"]

    def processed_raw(self, image: Image.Image, factors: Photometric | None = None):
        rgb = image.convert("RGB")  # deliberately no EXIF transpose
        key = f"{rgb.width}x{rgb.height}"
        if key not in self.config["layouts"]:
            raise ValueError(f"Unknown raw layout {key}; no automatic geometry rescaling")
        layout = self.config["layouts"][key]
        x = F.pil_to_tensor(rgb).to(torch.float32).div(255.0)
        x = x[:, :layout["processed_wh"][1], :].clone()
        valid = torch.ones(x.shape[-2:], dtype=torch.bool)
        for mask in layout["masks_xywh"]:
            left, top, width, height = mask["xywh"]
            valid[top:top + height, left:left + width] = False
        if factors is not None:
            if factors.brightness < 0 or factors.contrast < 0:
                raise ValueError("Photometric factors must be nonnegative")
            x[:, valid] = (x[:, valid] * factors.brightness).clamp(0, 1)
            weights = torch.tensor([0.2989, 0.587, 0.114], dtype=torch.float32)[:, None]
            gray_mean = (x[:, valid] * weights).sum(dim=0).mean()
            x[:, valid] = (x[:, valid] * factors.contrast + (1 - factors.contrast) * gray_mean).clamp(0, 1)
        x[:, ~valid] = self.mean[:, 0, 0, None]
        return x, valid, layout

    @staticmethod
    def resize(x, wh):
        return F.resize(x, [wh[1], wh[0]], interpolation=InterpolationMode.BILINEAR,
                        antialias=True)

    def letterbox(self, x):
        height, width = x.shape[-2:]
        tw, th = self.global_wh
        scale = min(tw / width, th / height)
        rw, rh = min(tw, max(1, half_up(width * scale))), min(th, max(1, half_up(height * scale)))
        left, top = (tw - rw) // 2, (th - rh) // 2
        canvas = self.mean.expand(3, th, tw).clone()
        canvas[:, top:top + rh, left:left + rw] = self.resize(x, (rw, rh))
        return canvas

    def normalize(self, x):
        return F.normalize(x, self.config["normalization"]["mean"], self.std, inplace=False)

    def __call__(self, image: Image.Image, factors: Photometric | None = None):
        raw, _, layout = self.processed_raw(image, factors)
        result = {"global": self.normalize(self.letterbox(raw))}
        if self.experiment == "e2":
            locals_ = []
            for left, top, width, height in layout["local_crops_xywh"]:
                crop = raw[:, top:top + height, left:left + width]
                locals_.append(self.normalize(self.resize(crop, (320, 320))))
            result["local"] = torch.stack(locals_)
        return result
