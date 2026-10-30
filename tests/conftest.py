import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture(scope="session")
def manifest():
    from src.pilot.dataset import FixedManifest
    return FixedManifest()
