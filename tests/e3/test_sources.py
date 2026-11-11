import pytest
import torch
from torch import nn


def test_exact_frozen_encoder_boundary_and_parent_train():
    from src.e3.sources import FrozenSource
    class Source(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 768)
            self.classifier = nn.Linear(768, 124)
        def encode(self, x):
            return self.encoder(x)
        def forward(self, x):
            return self.classifier(self.encode(x))
    source = Source()
    before = {k: v.clone() for k, v in source.state_dict().items()}
    adapter = FrozenSource(source, family="E1")
    adapter.train(True)
    features = adapter.encode({"global": torch.ones(2, 3)})
    assert features.shape == (2, 768) and not features.requires_grad
    assert not source.training and not any(p.requires_grad for p in adapter.parameters())
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in before.items())
    assert torch.equal(adapter({"global": torch.ones(2, 3)}), source.classifier(features))


def test_e2_global_then_batch_major_locals_and_mean_after_norm():
    from src.e3.sources import FrozenSource
    from src.e2.model import E2Model
    class Views(nn.Module):
        encode = E2Model.encode
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(1536, 124)
            self.calls = []
        def encode_view(self, x):
            markers = x[:, 0, 0, 0]
            self.calls.append(markers.tolist())
            return markers[:, None].square().expand(-1, 768)
    source = Views(); model = FrozenSource(source, family="E2")
    result = model.encode({"global": torch.tensor([2., 3.])[:, None, None, None].expand(2, 3, 1, 1),
                           "local": torch.arange(1., 9.).reshape(2, 4, 1, 1, 1).expand(2, 4, 3, 1, 1)})
    assert source.calls == [[2., 3.], [1., 2., 3., 4., 5., 6., 7., 8.]]
    assert result[0, 0] == 4 and result[0, 768] == 7.5
    assert result[1, 0] == 9 and result[1, 768] == 43.5


def test_test_or_root_escape_rejected_before_open(tmp_path, monkeypatch):
    from src.e3.data import ImageRows
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("image must not open"))
    with pytest.raises(ValueError):
        ImageRows([{"split": "test", "relative_path": "a/test.jpg", "label_id": 0}], tmp_path, "E1")
    with pytest.raises(ValueError):
        ImageRows([{"split": "train", "relative_path": "../escape.jpg", "label_id": 0}], tmp_path, "E1")[0]


def test_fixed_membership_and_test_lock():
    from src.e3.data import FixedManifest
    m = FixedManifest()
    assert len(m.records("train")) == 34997 and len(m.records("val")) == 7497
    assert set(m.n_train) == set(range(124)) and all(m.n_train.values())
    with pytest.raises(ValueError):
        m.records("test")


def test_selected_state_rejects_wrong_seed_epoch_kind_and_head():
    from src.e3.sources import validate_selected_state
    entry = {"family": "E1", "source_seed": 3001, "selected_epoch": 7}
    state = {"kind": "official_e1_checkpoint_v1", "role": "best", "config": {"seed": 3001},
             "progress": {"selected": {"epoch": 7}},
             "model": {"classifier.weight": torch.zeros(124, 768), "classifier.bias": torch.zeros(124)}}
    validate_selected_state(state, entry)
    import copy
    for mutate in [lambda s: s.update(role="last"), lambda s: s["config"].update(seed=2001),
                   lambda s: s["progress"]["selected"].update(epoch=8),
                   lambda s: s.update(kind="official_e2_checkpoint_v1"),
                   lambda s: s["model"].update({"classifier.weight": torch.zeros(124, 1536)})]:
        bad = copy.deepcopy(state); mutate(bad)
        with pytest.raises(ValueError):
            validate_selected_state(bad, entry)
