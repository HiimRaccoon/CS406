import hashlib

import pytest
import torch
from torch import nn
from torchvision.models import convnext_tiny
from torchvision.ops import StochasticDepth


def test_real_torchvision_architecture_head_init_groups_and_gradients(recipe):
    from src.e1.model import E1Model, build_optimizer
    backbone = convnext_tiny(weights=None, stochastic_depth_prob=.1)
    outer = torch.get_rng_state()
    m = E1Model(backbone, {**recipe["initialization"], "seed": 1001})
    assert torch.equal(outer, torch.get_rng_state())
    other = E1Model(backbone, {**recipe["initialization"], "seed": 1001})
    assert torch.equal(m.classifier.weight, other.classifier.weight)
    assert m.final_norm is backbone.classifier[0] and m.avgpool is backbone.avgpool
    depth = [b.p for b in m.modules() if isinstance(b, StochasticDepth)]
    assert depth[0] == 0 and depth[-1] == .1
    images = torch.randn(2, 3, 32, 32)  # Tiny mechanics, not target full-size acceptance.
    assert m.encode(images).shape == (2, 768)
    assert m(images).shape == (2, 124)
    m(images).sum().backward()
    assert next(m.features.parameters()).grad is not None
    optimizer, groups = build_optimizer(m, {**recipe["optimizer"], "lr": 1e-4})
    assert "classifier.weight" in groups["decay"] and "classifier.bias" in groups["no_decay"]
    scales = [name for name, _ in m.named_parameters() if name.endswith("layer_scale")]
    assert scales and all(name in groups["no_decay"] for name in scales)
    assert len({id(p) for g in optimizer.param_groups for p in g["params"]}) == len(list(m.parameters()))
    m.eval()
    assert torch.equal(m(images), m(images))


def test_digest_is_checked_in_full_before_loading(monkeypatch, tmp_path, recipe):
    import src.e1.model as module
    p = tmp_path / "convnext_tiny-983f1562.pth"
    p.write_bytes(b"synthetic, not pretrained")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Loaded unverified weights"))
    expected = recipe["model"]["pretrained"]["sha256"]
    with pytest.raises(ValueError):
        module.build_e1(p, expected, {**recipe["initialization"], "seed": 1001})
    with pytest.raises(ValueError):
        module.build_e1(p, "983f1562" + "0" * 56, {"seed": 1001})


def test_strict_verified_state_loading_is_preserved_at_external_edge(monkeypatch, tmp_path, recipe):
    import src.e1.model as module
    p = tmp_path / "convnext_tiny-983f1562.pth"
    backbone = convnext_tiny(weights=None, stochastic_depth_prob=.1)
    torch.save(backbone.state_dict(), p)
    # Replace only file-integrity external edge. No alternate digest can be configured in production.
    monkeypatch.setattr(module, "sha256_file", lambda _: recipe["model"]["pretrained"]["sha256"])
    m, metadata = module.build_e1(p, recipe["model"]["pretrained"]["sha256"], {**recipe["initialization"], "seed": 1001})
    assert torch.equal(next(m.features.parameters()), next(backbone.features.parameters()))
    bad = backbone.state_dict()
    bad.pop(next(iter(bad)))
    torch.save(bad, p)
    with pytest.raises(RuntimeError):
        module.build_e1(p, recipe["model"]["pretrained"]["sha256"], {**recipe["initialization"], "seed": 1001})


def test_unknown_parameter_category_rejected(recipe):
    from src.e1.model import build_optimizer
    m = nn.Module()
    m.mystery = nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError):
        build_optimizer(m, {**recipe["optimizer"], "lr": 1e-4})
