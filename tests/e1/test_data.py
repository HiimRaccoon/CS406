import dataclasses
from pathlib import Path

import pytest
import torch
from PIL import Image


def test_metadata_coverage_and_test_rejection_before_open(monkeypatch):
    from src.e1.data import FixedManifest
    m = FixedManifest()
    assert m.n_train == 34997 and len(m.records("val")) == 7497
    assert len({r.label_id for r in m.records("val")}) == 106
    assert (m.labels[0], m.labels[120], m.labels[123]) == ("Aland", "United States", "Vietnam")
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("Image opened for rejected split"))
    with pytest.raises(ValueError):
        m.records("test")


def test_changed_artifact_fails_before_reading_metadata(tmp_path):
    from src.e1.data import FixedManifest
    (tmp_path / "manifest.csv").write_text("changed")
    with pytest.raises(ValueError):
        FixedManifest(tmp_path)


@pytest.mark.parametrize("wh", [(1536, 662), (1030, 703)])
def test_both_layouts_global_only_deterministic_validation_and_path_guard(tmp_path, wh):
    from src.e1.data import E1Dataset, Record, resolve_image
    path = tmp_path / "Aland/a.jpg"
    path.parent.mkdir()
    Image.new("RGB", wh, (70, 120, 160)).save(path)
    record = Record("Aland/a.jpg", "Aland", 0, "val", "unused", *wh)
    class Mini:
        def records(self, split):
            if split != "val":
                raise ValueError("train/val only")
            return (record,)
    d = E1Dataset(Mini(), tmp_path, "val", 1001, {"train_enabled": False})
    x, y, rel = d[0]
    assert x.shape == (3, 384, 896) and y == 0 and rel == "Aland/a.jpg"
    assert torch.equal(x, d[0][0])
    assert (x[:, :3] == 0).all() if wh[0] == 1536 else (x[:, :, :161] == 0).all()
    with pytest.raises(ValueError):
        resolve_image(tmp_path, "../outside.jpg")
    with pytest.raises(ValueError):
        E1Dataset(Mini(), tmp_path, "test", 1001, {})
    with pytest.raises(ValueError):
        E1Dataset(Mini(), tmp_path, "val", 1001, {"train_enabled": True})


def test_epoch_order_is_complete_and_jitter_survives_persistent_workers(tmp_path, recipe):
    from src.e1.data import E1Dataset, Record, EpochSampler, make_loaders
    files = []
    for i in range(3):
        rel = f"Aland/{i}.jpg"
        p = tmp_path / rel
        p.parent.mkdir(exist_ok=True)
        Image.new("RGB", (1030, 703), (50 + i, 80, 130)).save(p)
        files.append(Record(rel, "Aland", 0, "train", "unused", 1030, 703))
    class Mini:
        def records(self, split):
            if split not in {"train", "val"}:
                raise ValueError("forbidden")
            return tuple(files)
    cfg = {"recipe": recipe, "seed": 1001}
    d = E1Dataset(Mini(), tmp_path, "train", 1001, recipe["preprocessing"]["augmentation"])
    sampler = EpochSampler(d, 1001)
    for epoch in (0, 1, 7):
        sampler.set_epoch(epoch)
        pairs = list(sampler)
        assert sorted(index for _, index in pairs) == [0, 1, 2]
        assert all(e == epoch for e, _ in pairs)
    assert not torch.equal(d[(0, 0)][0], d[(1, 0)][0])
    loaders = make_loaders(Mini(), tmp_path, cfg)
    for epoch in (0, 1):
        loaders["train"].sampler.set_epoch(epoch)
        batches = list(loaders["train"])
        assert [len(y) for _, y, _ in batches] == [2, 1]
        for xs, ys, paths in batches:
            for x, path in zip(xs, paths):
                assert torch.equal(x, d[(epoch, int(Path(path).stem))][0])
    # Restarted workers must recover the same sample/epoch values, not worker RNG draws.
    restarted = make_loaders(Mini(), tmp_path, cfg)
    restarted["train"].sampler.set_epoch(1)
    assert torch.equal(next(iter(restarted["train"]))[0], batches[0][0])
