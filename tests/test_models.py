import pytest
import torch
from torchvision.models import convnext_tiny
from torchvision.ops import StochasticDepth

from src.pilot.models import PilotModel, build_model, build_optimizer, initialize_classifier


@pytest.mark.parametrize("exp,dim", [("e1", 768), ("e1hr", 768), ("e2", 1536)])
def test_pilot_backbone_shapes_shared_weights_and_gradients(exp, dim):
    # Random-weight CPU mechanics, never pretrained quality or an official run.
    m = PilotModel(convnext_tiny(weights=None, stochastic_depth_prob=.1), exp,
                   {"method": "linear_default", "seed": 13})
    assert m.classifier.in_features == dim
    sd = [x for x in m.modules() if isinstance(x, StochasticDepth)]
    assert sd[0].p == 0 and sd[-1].p == pytest.approx(.1)
    views = {"global": torch.rand(2, 3, 32, 32)}
    if exp == "e2":
        views["local"] = torch.rand(2, 4, 3, 32, 32)
        assert sum(p.numel() for p in m.features.parameters()) < 30_000_000
    output = m(views)
    assert output.shape == (2, 124)
    torch.nn.functional.cross_entropy(output, torch.tensor([0, 1])).backward()
    assert next(m.features.parameters()).grad.abs().sum() > 0
    optimizer, names = build_optimizer(m, {"lr": .001, "betas": [.9, .999], "eps": 1e-8})
    assert "classifier.weight" in names["decay"] and "classifier.bias" in names["no_decay"]
    assert len(optimizer.param_groups) == 2


def test_head_init_rng_and_offline_integrity(tmp_path):
    a, b = torch.nn.Linear(768, 124), torch.nn.Linear(768, 124)
    state = torch.get_rng_state()
    for head in (a, b):
        initialize_classifier(head, {"method": "linear_default", "seed": 21})
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
    assert torch.equal(state, torch.get_rng_state())
    bad = tmp_path / "convnext_tiny-983f1562.pth"
    bad.write_bytes(b"not the pretrained weight file")
    with pytest.raises(ValueError, match="SHA-256"):
        build_model("e1", bad, {"method": "linear_default", "seed": 21})
