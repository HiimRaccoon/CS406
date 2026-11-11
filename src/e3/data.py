"""Frozen manifest membership and deterministic, train/validation-only image loading."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import ROOT, DIMENSIONS, contained, fingerprint, policy, sha256_file
from ..e1.data import FixedManifest as _FixedManifest, derived_seed
from ..preprocessing.geoguessr_v1 import Preprocessor


class FixedManifest:
    def __init__(self, artifacts_dir: Path = ROOT / "artifacts/baseline124"):
        self._manifest = _FixedManifest(artifacts_dir)
        self.labels = self._manifest.labels
        self.n_train = dict(self._manifest.n_train_by_label)
        self.fingerprints = self._manifest.fingerprints

    def records(self, split: str) -> tuple[dict, ...]:
        if split not in {"train", "val"}:
            raise ValueError("E3 image access is train/val only")
        return tuple(asdict(row) for row in self._manifest.records(split))


class ImageRows(Dataset):
    def __init__(self, records, data_root: Path, family: str):
        if family not in DIMENSIONS or any(r.get("split") not in {"train", "val"} for r in records):
            raise ValueError("Only fixed family/train/val images may be opened")
        self.records = tuple(records)
        if len({r["relative_path"] for r in records}) != len(records):
            raise ValueError("Duplicate image key")
        self.root = Path(data_root).resolve()
        self.processor = Preprocessor(family.lower())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if row["split"] not in {"train", "val"}:
            raise ValueError("Test access forbidden before image/hash access")
        path = contained(self.root, row["relative_path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError("Missing/corrupt fixed image")
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError("Current image layout differs from manifest")
            views = self.processor(image.convert("RGB"), factors=None)
        return views, row["label_id"], row["relative_path"]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_source_loader(manifest, *, source: dict, request: dict):
    split = request["split"]
    if split not in {"train", "val"}:
        raise ValueError("Test loader forbidden")
    stage = request["stage"]
    if stage not in {"acceptance_fixture", "final"}:
        raise ValueError("Explicit extraction RNG stage required")
    root_seed = request["root_seed"]
    if root_seed != source["source_seed"]:
        raise ValueError("Extraction root must match source seed")
    records = manifest.records(split)
    start = request.get("row_start", 0)
    if type(start) is not int or not 0 <= start <= len(records) or (start != len(records) and start % policy()["storage"]["rows_per_chunk"]):
        raise ValueError("Extraction resumes only from an uncommitted chunk start")
    generator = torch.Generator(device="cpu").manual_seed(derived_seed(root_seed, "E3-v1", stage, source["source_seed"], "loader", split))
    settings = policy()["storage"]["extraction"]
    return DataLoader(ImageRows(records[start:], request["data_root"], source["family"]),
                      batch_size=settings["original_image_batch_size"], shuffle=False, drop_last=False,
                      num_workers=settings["num_workers"], pin_memory=settings["pin_memory"],
                      persistent_workers=settings["persistent_workers"], prefetch_factor=settings["prefetch_factor"],
                      in_order=True, worker_init_fn=seed_worker, generator=generator)


def audit_train_val(manifest, data_root: Path) -> dict:
    rows = []
    for split in ("train", "val"):
        data = ImageRows(manifest.records(split), data_root, "E1")
        for i, record in enumerate(data.records):
            # Reuses the exact decode/layout/hash checks. No test rows reach this dataset.
            data[i]
            rows.append((split, record["relative_path"], record["sha256"]))
    return {"kind": "e3_current_data_audit_v1", "actual_root": str(Path(data_root).resolve()),
            "ordered_content_sha256": fingerprint(rows), "rows": len(rows),
            "fixed_metadata": manifest.fingerprints, "test_images_opened": 0}
