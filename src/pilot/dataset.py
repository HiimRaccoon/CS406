"""Read exact membership; test records are metadata only in this harness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from ..preprocessing.geoguessr_v1 import (REPO_ROOT, frozen_preprocessing, sha256_file,
                                         Photometric, Preprocessor)


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    label: str
    label_id: int
    split: str
    sha256: str
    width: int
    height: int


class FixedManifest:
    def __init__(self, artifacts: str | Path | None = None):
        self.preprocessing = frozen_preprocessing()
        self.root = Path(artifacts) if artifacts else REPO_ROOT / "artifacts/baseline124"
        expected = self.preprocessing["source_split"]["sha256"]
        self.fingerprints = {}
        for name, digest in expected.items():
            actual = sha256_file(self.root / name)
            if actual != digest:
                raise ValueError(f"Exact artifact hash mismatch: {name}")
            self.fingerprints[name] = actual
        self.summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        provenance = json.loads((self.root / "provenance.json").read_text(encoding="utf-8"))
        for name, digest in self.fingerprints.items():
            if provenance["files"][name]["sha256"] != digest:
                raise ValueError(f"Provenance mismatch: {name}")
        if provenance["source_commit"] != self.preprocessing["source_split"]["source_commit"]:
            raise ValueError("Source commit mismatch")
        for name, key in [("manifest.csv", "manifest_sha256"), ("labels.json", "labels_sha256"),
                          ("class_support.csv", "support_sha256")]:
            if self.summary[key] != expected[name]:
                raise ValueError(f"Original summary mismatch: {key}")
        self.labels = json.loads((self.root / "labels.json").read_text(encoding="utf-8"))
        if len(self.labels) != 124 or len(set(self.labels)) != 124:
            raise ValueError("Expected 124 unique labels")
        records = []
        paths, hashes = set(), set()
        with (self.root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                p = PurePosixPath(row["relative_path"])
                if p.is_absolute() or ".." in p.parts or ":" in str(p):
                    raise ValueError("Unsafe manifest path")
                r = ImageRecord(str(p), row["label"], int(row["label_id"]), row["split"],
                                row["sha256"], int(row["width"]), int(row["height"]))
                if not 0 <= r.label_id < 124 or self.labels[r.label_id] != r.label:
                    raise ValueError("Label ID mismatch")
                if r.split not in {"train", "val", "test"}:
                    raise ValueError("Unknown split")
                if f"{r.width}x{r.height}" not in self.preprocessing["layouts"]:
                    raise ValueError("Unknown raw layout")
                if r.relative_path in paths or r.sha256 in hashes:
                    raise ValueError("Duplicate membership/image identity")
                paths.add(r.relative_path)
                hashes.add(r.sha256)
                records.append(r)
        self.records = tuple(records)
        self.counts = Counter(r.split for r in records)
        if dict(self.counts) != self.preprocessing["source_split"]["expected_counts"]:
            raise ValueError("Fixed split counts mismatch")
        for split, n in self.summary["class_coverage"].items():
            if len({r.label_id for r in records if r.split == split}) != n:
                raise ValueError("Class coverage mismatch")
        support = Counter((r.label_id, r.split) for r in records)
        with (self.root / "class_support.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 124:
            raise ValueError("Class support table length mismatch")
        for row in rows:
            if self.labels[int(row["label_id"])] != row["label"]:
                raise ValueError("Class support label mismatch")
            for split in ("train", "val", "test"):
                if support[int(row["label_id"]), split] != int(row[split]):
                    raise ValueError("Class support count mismatch")

    def for_split(self, split: str) -> tuple[ImageRecord, ...]:
        if split not in {"train", "val"}:
            raise ValueError("Pilot image access is restricted to train/val; test is metadata only")
        return tuple(r for r in self.records if r.split == split)

    def verify_images(self, data_root: Path, splits=("train", "val")) -> dict:
        counts = {}
        for split in splits:
            records = self.for_split(split)  # rejects test before opening any test image
            for r in records:
                if sha256_file(resolve_image(data_root, r.relative_path)) != r.sha256:
                    raise ValueError(f"Local image hash mismatch: {r.relative_path}")
            counts[split] = len(records)
        return counts


def resolve_image(data_root: Path, relative_path: str) -> Path:
    root = Path(data_root).resolve()
    p = (root / relative_path).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Image path escapes dataset root")
    if not p.is_file():
        raise FileNotFoundError(f"Missing image: {p}; dataset root must contain label folders")
    return p


def derived_seed(seed: int, *parts) -> int:
    text = "|".join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


class PilotDataset(Dataset):
    def __init__(self, manifest: FixedManifest, data_root, split, processor: Preprocessor,
                 seed: int, augmentation=None):
        self.records = manifest.for_split(split)
        self.data_root, self.split, self.processor, self.seed = data_root, split, processor, seed
        self.augmentation = augmentation or {"enabled": False}
        if split != "train" and self.augmentation["enabled"]:
            raise ValueError("Validation augmentation is forbidden")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        r = self.records[index]
        factors = None
        if self.augmentation["enabled"]:
            generator = torch.Generator().manual_seed(derived_seed(self.seed, epoch, r.relative_path, "jitter"))
            values = []
            for key in ("brightness", "contrast"):
                amplitude = self.augmentation[key]
                values.append(1 - amplitude + 2 * amplitude * torch.rand((), generator=generator).item())
            factors = Photometric(*values)
        with Image.open(resolve_image(self.data_root, r.relative_path)) as image:
            views = self.processor(image, factors)
        return views, r.label_id


class EpochSampler(Sampler):
    """Epoch travels with each index, including through persistent worker processes."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(derived_seed(self.seed, self.epoch, "order"))
        for index in torch.randperm(len(self.dataset), generator=generator).tolist():
            yield self.epoch, index

    def __len__(self):
        return len(self.dataset)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_loader(dataset, cfg, *, train):
    runtime = cfg["runtime"]
    generator = torch.Generator().manual_seed(derived_seed(cfg["seed"], "loader", dataset.split))
    kwargs = dict(batch_size=cfg["batch"]["microbatch"], num_workers=runtime["num_workers"],
                  pin_memory=runtime["pin_memory"], persistent_workers=runtime["persistent_workers"],
                  drop_last=False, generator=generator, worker_init_fn=seed_worker)
    if runtime["num_workers"]:
        kwargs["prefetch_factor"] = runtime["prefetch_factor"]
    sampler = EpochSampler(dataset, cfg["seed"]) if train else None
    return DataLoader(dataset, sampler=sampler, shuffle=False, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--verify-images", nargs="+", choices=["train", "val"])
    args = parser.parse_args()
    m = FixedManifest()
    report = {"status": "pass", "counts": dict(m.counts), "labels": len(m.labels),
              "fingerprints": m.fingerprints, "test_images_opened": 0}
    if args.verify_images:
        if args.data_root is None:
            parser.error("--verify-images requires --data-root")
        report["image_hash_checks"] = m.verify_images(args.data_root, args.verify_images)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
