"""Opt-in real CUDA/offline-weight mechanics; no dataset, quality score or Slurm submission."""
import os

import pytest
import torch

from src.pilot.models import build_model


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("CS406_GPU_SMOKE") != "1", reason="explicit opt-in required")
def test_cuda_offline_pretrained_forward_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    path = os.environ["CS406_WEIGHTS_PATH"]
    m, metadata = build_model("e2", path, {"method": "linear_default", "seed": 101},
                              os.environ["CS406_EXPECTED_WEIGHT_SHA256"])
    # Synthetic diagnostic initialization seed only, not the research seed protocol.
    m.cuda().train()
    views = {"global": torch.rand(1, 3, 384, 896, device="cuda"),
             "local": torch.rand(1, 4, 3, 320, 320, device="cuda")}
    output = m(views)
    torch.nn.functional.cross_entropy(output, torch.tensor([0], device="cuda")).backward()
    assert output.shape == (1, 124) and len(metadata["sha256"]) == 64
