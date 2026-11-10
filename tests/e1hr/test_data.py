import dataclasses
from pathlib import Path

import pytest
import torch
from PIL import Image


def test_metadata_coverage_and_test_rejection_before_open(monkeypatch):
    from src.e1hr.data import FixedManifest
    m = FixedManifest()
    assert m.n_train == 34997 and len(m.records("val")) == 7497
    assert len({r.label_id for r in m.records("val")}) == 106
    assert (m.labels[0], m.labels[120], m.labels[123]) == ("Aland", "United States", "Vietnam")
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("Image opened for rejected split"))
    with pytest.raises(ValueError):
        m.records("test")


def test_changed_artifact_fails_before_reading_metadata(tmp_path):
    from src.e1hr.data import FixedManifest
    (tmp_path / "manifest.csv").write_text("changed")
    with pytest.raises(ValueError):
        FixedManifest(tmp_path)


@pytest.mark.parametrize("wh", [(1536, 662), (1030, 703)])
def test_both_layouts_global_only_deterministic_validation_and_path_guard(tmp_path, wh):
    from src.e1hr.data import E1HRDataset, Record, resolve_image
    path = tmp_path / "Aland/a.jpg"
    path.parent.mkdir()
    Image.new("RGB", wh, (70, 120, 160)).save(path)
    record = Record("Aland/a.jpg", "Aland", 0, "val", "unused", *wh)
    class Mini:
        def records(self, split):
            if split != "val":
                raise ValueError("train/val only")
            return (record,)
    d = E1HRDataset(Mini(), tmp_path, "val", 1001, {"train_enabled": False})
    x, y, rel = d[0]
    assert x.shape == (3, 576, 1344) and y == 0 and rel == "Aland/a.jpg"
    assert torch.equal(x, d[0][0])
    assert (x[:, :4] == 0).all() if wh[0] == 1536 else (x[:, :, :241] == 0).all()
    with pytest.raises(ValueError):
        resolve_image(tmp_path, "../outside.jpg")
    with pytest.raises(ValueError):
        E1HRDataset(Mini(), tmp_path, "test", 1001, {})
    with pytest.raises(ValueError):
        E1HRDataset(Mini(), tmp_path, "val", 1001, {"train_enabled": True})


@pytest.mark.filterwarnings("ignore:.*pin_memory.*no accelerator.*:UserWarning")
def test_epoch_order_is_complete_and_jitter_survives_persistent_workers(tmp_path, recipe):
    from src.e1hr.data import E1HRDataset, Record, EpochSampler, make_loaders
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
    d = E1HRDataset(Mini(), tmp_path, "train", 1001, recipe["preprocessing"]["augmentation"])
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
        if epoch == 0:
            boundary_generator = loaders["train"].generator.get_state().clone()
        assert [len(y) for _, y, _ in batches] == [2, 1]
        for xs, ys, paths in batches:
            for x, path in zip(xs, paths):
                assert torch.equal(x, d[(epoch, int(Path(path).stem))][0])
    # Restarted workers must recover the same sample/epoch values, not worker RNG draws.
    restarted = make_loaders(Mini(), tmp_path, cfg)
    restarted["train"].generator.set_state(boundary_generator)
    restarted["train"].prepare_resume(1)
    restarted["train"].sampler.set_epoch(1)
    assert torch.equal(next(iter(restarted["train"]))[0], batches[0][0])
    assert torch.equal(restarted["train"].generator.get_state(), loaders["train"].generator.get_state())


@pytest.mark.parametrize('parts,expected', [
    ((1001,0,'order'),8021788208606133414),
    ((1001,0,'Aland/synthetic.jpg','jitter'),1711155315711192502),
    ((1001,'loader','train'),381993971064482208),
    ((1001,'loader','val'),5335367272077000490),
])
def test_inherited_seed_encoding_without_architecture_token(parts,expected):
    from src.e1hr.data import derived_seed
    assert derived_seed(*parts)==expected


@pytest.mark.parametrize('wh,content,pad', [
    ((1536,662),(1344,567),(0,4,0,5)),
    ((1030,703),(861,576),(241,0,242,0)),
])
def test_direct_raw_hr_and_independent_padding(tmp_path,wh,content,pad):
    import numpy as np
    from src.e1hr.data import E1HRDataset, Record
    from src.preprocessing.geoguessr_v1 import Preprocessor
    p=tmp_path/'Aland/textured.png'; p.parent.mkdir()
    pixels=np.random.default_rng(44).integers(0,256,(wh[1],wh[0],3),dtype=np.uint8)
    Image.fromarray(pixels).save(p)
    class Mini:
        def records(self,split): return (Record('Aland/textured.png','Aland',0,'val','unused',*wh),)
    d=E1HRDataset(Mini(),tmp_path,'val',1001,{'train_enabled':False})
    x,_,_=d[0]
    with Image.open(p) as image:
        processor=Preprocessor('e1hr',global_wh=(1344,576))
        assert torch.equal(x,processor(image)['global'])
        wrong=Preprocessor('e1')(image)['global']
    wrong=torch.nn.functional.interpolate(wrong[None],size=(576,1344),mode='bilinear',align_corners=False,antialias=True)[0]
    assert not torch.allclose(x,wrong)
    left,top,right,bottom=pad
    assert content==(1344-left-right,576-top-bottom)
    if left: assert (x[:,:,:left]==0).all()
    if top: assert (x[:,:top,:]==0).all()
    if right: assert (x[:,:,-right:]==0).all()
    if bottom: assert (x[:,-bottom:,:]==0).all()


def test_audit_rejects_test_before_any_byte_open(monkeypatch,tmp_path):
    from src.e1hr.data import FixedManifest
    m=FixedManifest()
    monkeypatch.setattr('src.e1hr.data.sha256_file',lambda *a:pytest.fail('bytes opened before rejecting test'))
    with pytest.raises(ValueError): m.verify_images(tmp_path,('train','test'))


def test_missing_corrupt_and_unknown_layout_fail(tmp_path):
    from src.e1hr.data import E1HRDataset,Record
    class Mini:
        def records(self,split): return (Record('Aland/a.png','Aland',0,'val','unused',1030,703),)
    d=E1HRDataset(Mini(),tmp_path,'val',1001,{'train_enabled':False})
    with pytest.raises(FileNotFoundError): d[0]
    p=tmp_path/'Aland/a.png'; p.parent.mkdir();p.write_bytes(b'corrupt')
    with pytest.raises(OSError): d[0]
    Image.new('RGB',(40,40)).save(p)
    with pytest.raises(ValueError): d[0]
