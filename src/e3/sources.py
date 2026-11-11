"""Selected complete source models, frozen at their unchanged encode boundary."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .config import (DIMENSIONS, ROOT, fingerprint, read_json, registry, sha256_file,
                     validate_source_binding, verify_payload)


class FrozenSource(nn.Module):
    def __init__(self, source: nn.Module, *, family: str):
        super().__init__()
        if family not in DIMENSIONS:
            raise ValueError("Unknown source family")
        self.source, self.family = source, family
        self.source.requires_grad_(False)
        self.train(False)

    @property
    def original_head(self):
        return self.source.classifier

    def train(self, mode=True):
        super().train(False)
        self.source.eval()
        return self

    @torch.no_grad()
    def encode(self, views):
        self.source.eval()
        value = self.source.encode(views["global"] if self.family == "E1" else views)
        if (value.ndim != 2 or value.shape[1] != DIMENSIONS[self.family]
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("Invalid unmodified preclassifier features")
        return value

    @torch.no_grad()
    def forward(self, views):
        self.source.eval()
        return self.source(views["global"] if self.family == "E1" else views)


def validate_selected_state(state: dict, entry: dict) -> None:
    try:
        dimension = DIMENSIONS[entry["family"]]
        if (state["kind"] != f"official_{entry['family'].lower()}_checkpoint_v1"
                or state["role"] != "best" or state["config"]["seed"] != entry["source_seed"]
                or state["progress"]["selected"]["epoch"] != entry["selected_epoch"]
                or tuple(state["model"]["classifier.weight"].shape) != (124, dimension)
                or tuple(state["model"]["classifier.bias"].shape) != (124,)):
            raise ValueError("Source selected state/seed/epoch/head mismatch")
        if any(not torch.isfinite(v).all() for v in state["model"].values() if torch.is_floating_point(v)):
            raise ValueError("Nonfinite source tensors")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete selected source envelope") from exc


def _authorized_source(entry, request):
    # Stage inspection is intentionally before any model or binary deserialization.
    from .runtime import preflight
    accepted = preflight(request, operation=request["operation"])
    declaration = accepted["declaration"]
    entries = declaration.get("sources", [declaration.get("source")])
    if entry not in entries:
        raise ValueError("Source is outside exact approved declaration")
    validate_source_binding(entry, accepted="source_acceptance_sha256" in entry)
    return accepted


def verify_source(entry: dict, contract: dict) -> dict:
    _authorized_source(entry, contract)
    for name in ("binary", "report"):
        path = Path(entry[f"actual_{name}_path"])
        if not path.is_file() or sha256_file(path) != entry[f"{name}_sha256"]:
            raise ValueError(f"Selected source {name} hash mismatch")
    report = read_json(entry["actual_report_path"])
    expected_kind = f"{entry['family'].lower()}_three_seed_validation_report_v1"
    if report.get("kind") != expected_kind:
        raise ValueError("Historical report schema/kind mismatch")
    return {"source": entry, "report_kind": expected_kind, "binary_verified": True}


def load_frozen_source(entry: dict, *, contract: dict, device):
    verification = verify_source(entry, contract)
    # Trusted, explicitly authorized selected binaries only; the full historical envelope
    # contains RNG/optimizer objects, so weights_only=False is necessary after byte pins.
    state = torch.load(entry["actual_binary_path"], map_location="cpu", weights_only=False)
    validate_selected_state(state, entry)
    if state["config"].get("contract_sha256") != entry["source_contract_sha256"]:
        raise ValueError("Source checkpoint contract mismatch")
    from torchvision.models import convnext_tiny
    from ..e1.model import E1Model
    from ..e2.model import E2Model
    with torch.random.fork_rng(devices=[]):
        backbone = convnext_tiny(weights=None, stochastic_depth_prob=.1)
        cls = E1Model if entry["family"] == "E1" else E2Model
        source = cls(backbone, {"method": "linear_default", "seed": entry["source_seed"]})
    source.load_state_dict(state["model"], strict=True)
    frozen = FrozenSource(source, family=entry["family"]).to(device)
    return frozen, verification
