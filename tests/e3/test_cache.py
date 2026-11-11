import copy
import pytest
import torch


def test_replay_tolerance_never_waives_prediction_change():
    from src.e3.cache import compare_replay
    from src.e3.config import policy
    ref = torch.zeros(2, 124)
    other = ref.clone(); other[0, 1] = 1e-7
    with pytest.raises(ValueError, match="discrete"):
        compare_replay(ref, other, torch.tensor([0, 1]), policy()["replay"])
    assert compare_replay(ref, ref.clone(), torch.tensor([0, 1]), policy()["replay"])["discrete_equal"]


def test_cache_requires_complete_ordered_manifest(tmp_path):
    from src.e3.cache import _build_payloads, _validate_payloads
    rows = [{"relative_path": f"A/{i}.jpg", "label_id": i % 2} for i in range(5)]
    values = torch.arange(5 * 768, dtype=torch.float32).reshape(5, 768)
    build = _build_payloads(tmp_path, rows, [(values, torch.tensor([0,1,0,1,0]), [r["relative_path"] for r in rows])],
                            dimension=768, identity={"source": "first"})
    _validate_payloads(tmp_path, build, rows, dimension=768)
    assert not (tmp_path / "index.json").exists()
    bad = copy.deepcopy(rows); bad.reverse()
    with pytest.raises(ValueError):
        _validate_payloads(tmp_path, build, bad, dimension=768)
    data = tmp_path / build["chunks"][0]["payload"]["relative_path"]
    data.write_bytes(data.read_bytes()[:-4])
    with pytest.raises(ValueError):
        _validate_payloads(tmp_path, build, rows, dimension=768)


def test_seed_or_fusion_swap_rejected_and_committed_chunks_reused(tmp_path):
    from src.e3.cache import _build_payloads
    rows = [{"relative_path": "A/0.jpg", "label_id": 0}]
    batch = [(torch.ones(1,1536),torch.tensor([0]),["A/0.jpg"])]
    first = _build_payloads(tmp_path, rows, batch, dimension=1536, identity={"source": "seed2001"})
    reused = _build_payloads(tmp_path, rows, [], dimension=1536, identity={"source": "seed2001"})
    assert reused == first
    with pytest.raises(ValueError):
        _build_payloads(tmp_path, rows, batch, dimension=1536, identity={"source": "seed3001"})
    with pytest.raises(ValueError):
        _build_payloads(tmp_path, rows, batch, dimension=768, identity={"source": "seed2001"})


def test_partial_index_cannot_publish(tmp_path):
    from src.e3.cache import _build_payloads
    rows = [{"relative_path": f"A/{i}.jpg", "label_id": 0} for i in range(3)]
    with pytest.raises(ValueError):
        _build_payloads(tmp_path, rows, [(torch.ones(2,768),torch.tensor([0,0]),["A/0.jpg","A/1.jpg"])],
                        dimension=768, identity={"source":"first"})
    assert not (tmp_path / "index.json").exists()
    assert not list((tmp_path / "chunks").glob("commit_*.json"))


def test_atomic_failure_preserves_old_bytes(tmp_path, monkeypatch):
    from src.e3.io import atomic_json
    import os
    p = tmp_path / "state.json"; p.write_text('{"old":true}\n')
    def failed(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(os, "replace", failed)
    with pytest.raises(OSError):
        atomic_json(p, {"new": True})
    assert p.read_text() == '{"old":true}\n'
    assert not list(tmp_path.glob("*.tmp.*"))


@pytest.mark.parametrize("corruption",["duplicate_json_key","trailing_npy_bytes"])
def test_rehashed_payload_still_rejects_invalid_wire_format(tmp_path,corruption):
    from src.e3.cache import _build_payloads,_validate_payloads
    from src.e3.config import payload
    rows=[{"relative_path":"A/0.jpg","label_id":0}]
    build=_build_payloads(tmp_path,rows,[(torch.ones(1,768),torch.tensor([0]),["A/0.jpg"])],dimension=768,identity={"source":"nonofficial"})
    if corruption=="duplicate_json_key":
        path=tmp_path/"rows.jsonl"
        path.write_text('{"row_index":0,"relative_path":"A/0.jpg","label_id":0,"label_id":0}\n')
        build["row_index"]=payload(path,tmp_path)
    else:
        path=tmp_path/build["chunks"][0]["payload"]["relative_path"]
        path.write_bytes(path.read_bytes()+b"unlisted trailing bytes")
        build["chunks"][0]["payload"]=payload(path,tmp_path)
    with pytest.raises(ValueError): _validate_payloads(tmp_path,build,rows,dimension=768)


def test_dataset_checks_cache_acceptance_before_any_feature_validation(tmp_path,monkeypatch):
    from src.e3 import cache
    path=tmp_path/"index.json"; path.write_text('{}')
    monkeypatch.setattr(cache,"validate_cache",lambda *a,**k:pytest.fail("unaccepted features must not be consumed"))
    def rejected(*args,**kwargs): raise ValueError("wrong stage receipt")
    monkeypatch.setattr(cache,"require_approval",rejected)
    with pytest.raises(ValueError,match="wrong stage"):
        cache.CacheDataset(path,expected={"family":"E1","source_seed":2001,"split":"val","cache_acceptance_receipt":tmp_path/"wrong.json"})
