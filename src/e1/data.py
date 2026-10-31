"""Exact train/validation membership, frozen views, and sample-keyed randomness."""
from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import ROOT, load_recipe, read_json, sha256_file
from ..preprocessing.geoguessr_v1 import Photometric, Preprocessor, frozen_preprocessing


@dataclass(frozen=True)
class Record:
    relative_path: str
    label: str
    label_id: int
    split: str
    sha256: str
    width: int
    height: int


def _safe_relative(value: str) -> PurePosixPath:
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or ":" in value or "\\" in value or p.as_posix() != value or len(p.parts) != 2:
        raise ValueError("Unsafe or noncanonical manifest path")
    return p


class FixedManifest:
    def __init__(self, artifacts_dir: Path = ROOT / "artifacts/baseline124"):
        root = Path(artifacts_dir)
        fixed = load_recipe()["data"]
        frozen_preprocessing()
        self.fingerprints = {}
        for name, expected in fixed["artifacts"].items():
            p = root / name
            if (not p.is_file() or p.stat().st_size != expected["bytes"]
                    or sha256_file(p) != expected["sha256"]):
                raise ValueError(f"Fixed artifact hash/length mismatch: {name}")
            self.fingerprints[name] = expected["sha256"]
        provenance = read_json(root / "provenance.json")
        if provenance["source_commit"] != fixed["source_commit"] or provenance["files"] != fixed["artifacts"]:
            raise ValueError("Split provenance mismatch")
        summary = read_json(root / "summary.json")
        for name, key in [("manifest.csv", "manifest_sha256"), ("labels.json", "labels_sha256"), ("class_support.csv", "support_sha256")]:
            if summary[key] != self.fingerprints[name]:
                raise ValueError("Summary fingerprint mismatch")
        self.labels = read_json(root / "labels.json")
        if len(self.labels) != 124 or len(set(self.labels)) != 124:
            raise ValueError("Invalid 124-label mapping")
        rows, paths, hashes, groups = [], set(), set(), set()
        with (root / "manifest.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                p = _safe_relative(row["relative_path"])
                r = Record(row["relative_path"], row["label"], int(row["label_id"]), row["split"],
                           row["sha256"], int(row["width"]), int(row["height"]))
                if (not 0 <= r.label_id < 124 or self.labels[r.label_id] != r.label or p.parts[0] != r.label
                        or r.split not in {"train", "val", "test"} or (r.width, r.height) not in {(1536, 662), (1030, 703)}
                        or not re.fullmatch("[0-9a-f]{64}", r.sha256)):
                    raise ValueError("Invalid manifest record")
                if r.relative_path in paths or r.sha256 in hashes or row["group_id"] in groups:
                    raise ValueError("Duplicate fixed membership")
                paths.add(r.relative_path)
                hashes.add(r.sha256)
                groups.add(row["group_id"])
                rows.append(r)
        counts = dict(Counter(r.split for r in rows))
        coverage = {s: len({r.label_id for r in rows if r.split == s}) for s in counts}
        if counts != fixed["split_counts"] or counts != summary["split_counts"] or coverage != fixed["supported_labels"] or coverage != summary["class_coverage"]:
            raise ValueError("Split count/coverage mismatch")
        support = Counter((r.label_id, r.split) for r in rows)
        with (root / "class_support.csv").open(encoding="utf-8", newline="") as stream:
            classes = list(csv.DictReader(stream))
        if len(classes) != 124 or {int(r["label_id"]) for r in classes} != set(range(124)):
            raise ValueError("Invalid support coverage")
        self.n_train = counts["train"]
        self.n_train_by_label = {}
        for row in classes:
            i = int(row["label_id"])
            if row["label"] != self.labels[i] or any(int(row[s]) != support[i, s] for s in counts):
                raise ValueError("Class support mismatch")
            self.n_train_by_label[i] = int(row["train"])
        self._records = {s: tuple(r for r in rows if r.split == s) for s in ("train", "val")}

    def records(self, split: str) -> tuple[Record, ...]:
        if split not in {"train", "val"}:
            raise ValueError("E1 image access is train/val only; test is metadata only")
        return self._records[split]

    def verify_images(self, data_root: Path, splits=("train", "val")) -> dict:
        selected = [(s, self.records(s)) for s in splits]
        for _, records in selected:
            for r in records:
                if sha256_file(resolve_image(data_root, r.relative_path)) != r.sha256:
                    raise ValueError(f"Image hash mismatch: {r.relative_path}")
        return {s: len(records) for s, records in selected}


def resolve_image(data_root: Path, relative_path: str) -> Path:
    _safe_relative(relative_path)
    root = Path(data_root).resolve()
    p = (root / relative_path).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Image escapes dataset root")
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def derived_seed(seed: int, *parts) -> int:
    value = "|".join(map(str, (seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


class E1Dataset(Dataset):
    def __init__(self, manifest, data_root, split, seed, augmentation):
        if split not in {"train", "val"}:
            raise ValueError("E1 image access is train/val only")
        self.records = manifest.records(split)
        self.data_root, self.split, self.seed = Path(data_root), split, seed
        self.augmentation = dict(augmentation)
        if split == "val" and self.augmentation.get("train_enabled", False):
            raise ValueError("Validation augmentation is forbidden")
        self.processor = Preprocessor("e1")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        epoch, index = index if isinstance(index, tuple) else (0, index)
        if type(epoch) is not int or not 0 <= epoch < 8:
            raise ValueError("Epoch outside original E1 budget")
        r = self.records[index]
        factors = None
        if self.augmentation.get("train_enabled", False):
            g = torch.Generator().manual_seed(derived_seed(self.seed, epoch, r.relative_path, "jitter"))
            values = [1 - self.augmentation[k] + 2 * self.augmentation[k] * torch.rand((), generator=g).item()
                      for k in ("brightness_amplitude", "contrast_amplitude")]
            factors = Photometric(*values)
        with Image.open(resolve_image(self.data_root, r.relative_path)) as image:
            if image.size != (r.width, r.height):
                raise ValueError("Decoded dimensions differ from manifest")
            tensor = self.processor(image, factors)["global"]
        return tensor, r.label_id, r.relative_path


class EpochSampler(Sampler):
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch):
        if type(epoch) is not int or not 0 <= epoch < 8:
            raise ValueError("Epoch outside E1 budget")
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator().manual_seed(derived_seed(self.seed, self.epoch, "order"))
        return iter((self.epoch, i) for i in torch.randperm(len(self.dataset), generator=g).tolist())

    def __len__(self):
        return len(self.dataset)


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


class E1Loader(DataLoader):
    """Separate restarted worker bootstrap from committed loader RNG progression."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.generator is None:
            raise ValueError("E1 loaders require an explicit independent generator")
        self._worker_bootstrap_state = self.generator.get_state().clone()
        self._resume_bootstrap = False

    def prepare_resume(self, completed_epoch):
        if self._iterator is not None:
            raise ValueError("Restore into fresh loaders, not live persistent workers")
        self._resume_bootstrap = bool(completed_epoch and self.persistent_workers and self.num_workers)

    def __iter__(self):
        if not self._resume_bootstrap:
            return super().__iter__()
        committed = self.generator.get_state()
        # Recreate the original worker base seeds without adding a logical loader draw.
        # All sample order/jitter remains independently keyed by run/epoch/path.
        self.generator.set_state(self._worker_bootstrap_state)
        try:
            iterator = super().__iter__()
        finally:
            self.generator.set_state(committed)
        self._resume_bootstrap = False
        return iterator


def make_loaders(manifest, data_root: Path, config: dict) -> dict:
    recipe, seed = config["recipe"], config["seed"]
    runtime, batch = recipe["runtime"], recipe["batch"]
    result = {}
    for split in ("train", "val"):
        augmentation = recipe["preprocessing"]["augmentation"] if split == "train" else {"train_enabled": False}
        dataset = E1Dataset(manifest, data_root, split, seed, augmentation)
        generator = torch.Generator().manual_seed(derived_seed(seed, "loader", split))
        kwargs = dict(batch_size=batch["microbatch"], num_workers=runtime["num_workers"],
                      pin_memory=runtime["pin_memory"], persistent_workers=runtime["persistent_workers"],
                      drop_last=False, generator=generator, worker_init_fn=seed_worker, shuffle=False)
        if runtime["num_workers"]:
            kwargs["prefetch_factor"] = runtime["prefetch_factor"]
        result[split] = E1Loader(dataset, sampler=EpochSampler(dataset, seed) if split == "train" else None, **kwargs)
    return result
