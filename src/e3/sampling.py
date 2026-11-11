"""Independent CPU streams for complete permutations and class-then-image draws."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.utils.data import Sampler

from .config import fingerprint, policy as approved_policy, validate_record, verify_payload
from ..e1.data import derived_seed


class DrawSampler(Sampler):
    def __init__(self, rows, *, policy: str, root_seed: int, source_seed: int, stage: str):
        if policy not in {"natural", "crt"} or stage not in {"pilot", "final", "acceptance_fixture"}:
            raise ValueError("Explicit sampler policy and RNG stage required")
        if type(root_seed) is not int or type(source_seed) is not int or not rows:
            raise ValueError("Explicit integer seeds/nonempty rows required")
        self.rows = tuple(rows)
        self.policy, self.root_seed, self.source_seed, self.stage = policy, root_seed, source_seed, stage
        self.class_indices = {c: tuple(i for i, row in enumerate(rows) if row["label_id"] == c) for c in range(124)}
        if policy == "crt" and any(not indices for indices in self.class_indices.values()):
            raise ValueError("cRT retains all 124 positive-support train classes")
        self.identity = fingerprint({"rows": self.rows, "policy": policy, "root_seed": root_seed,
                                     "source_seed": source_seed, "stage": stage})
        self.set_window(0)

    def set_window(self, index: int):
        if type(index) is not int or index < 0:
            raise ValueError("Window index must be nonnegative integer")
        self.window, self.cursor = index, 0
        def generator(tag):
            return torch.Generator(device="cpu").manual_seed(derived_seed(
                self.root_seed, "E3-v1", self.stage, self.source_seed, "sampler", tag, index))
        self.natural_generator = generator("natural")
        self.class_generator = generator("crt-class")
        self.image_generator = generator("crt-image")
        self.permutation = torch.randperm(len(self.rows), generator=self.natural_generator, device="cpu") if self.policy == "natural" else None

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        while self.cursor < len(self.rows):
            if self.policy == "natural":
                row = int(self.permutation[self.cursor])
            else:
                label = int(torch.randint(124, (1,), generator=self.class_generator, device="cpu"))
                indices = self.class_indices[label]
                within_class = int(torch.randint(len(indices), (1,), generator=self.image_generator, device="cpu"))
                row = indices[within_class]
            self.cursor += 1
            yield row

    def state_dict(self) -> dict:
        return copy.deepcopy({"identity": self.identity, "policy": self.policy, "stage": self.stage,
                              "root_seed": self.root_seed, "source_seed": self.source_seed,
                              "window": self.window, "cursor": self.cursor, "permutation": self.permutation,
                              "natural_rng": self.natural_generator.get_state(),
                              "class_rng": self.class_generator.get_state(), "image_rng": self.image_generator.get_state()})

    def load_state_dict(self, state: dict):
        try:
            if (state["identity"] != self.identity or state["policy"] != self.policy
                    or state["stage"] != self.stage or state["root_seed"] != self.root_seed
                    or state["source_seed"] != self.source_seed or type(state["cursor"]) is not int
                    or not 0 <= state["cursor"] <= len(self.rows)):
                raise ValueError("Sampler source/stream/cursor mismatch")
            self.set_window(state["window"])
            if self.policy == "natural":
                p = state["permutation"]
                if (p.dtype != torch.int64 or tuple(p.shape) != (len(self.rows),)
                        or not torch.equal(torch.sort(p).values, torch.arange(len(self.rows)))
                        or not torch.equal(p, self.permutation)):
                    raise ValueError("Invalid or foreign natural permutation")
                self.permutation = p.clone()
            elif state["permutation"] is not None:
                raise ValueError("cRT cannot consume a natural permutation")
            self.cursor = state["cursor"]
            self.natural_generator.set_state(state["natural_rng"])
            self.class_generator.set_state(state["class_rng"])
            self.image_generator.set_state(state["image_rng"])
        except (KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("Invalid serialized sampler state") from exc


def make_sampler(cache_index: dict, *, policy: str, seed: int, rng_contract: dict, cache_root: Path):
    validate_record(cache_index, "cache_index")
    if cache_index["split"] != "train" or cache_index["complete"] is not True:
        raise ValueError("Sampling requires a complete fixed train cache")
    if (rng_contract["policy"] != approved_policy()["rng"] or seed != rng_contract["root_seed"]
            or cache_index["source_seed"] != rng_contract["source_seed"]):
        raise ValueError("Unreviewed RNG policy or source/root bindings")
    rows = [json.loads(line) for line in verify_payload(cache_index["row_index"], cache_root).read_text(encoding="utf-8").splitlines()]
    if len(rows) != 34997 or cache_index["row_count"] != 34997:
        raise ValueError("Complete train membership required")
    return DrawSampler(rows, policy=policy, root_seed=seed, source_seed=rng_contract["source_seed"], stage=rng_contract["stage"])
