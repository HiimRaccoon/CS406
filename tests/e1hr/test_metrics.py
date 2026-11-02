import pytest
import torch


def test_cross_bucket_f1_retains_false_positives_and_unsupported_outputs():
    from src.e1hr.metrics import Metrics124
    metric = Metrics124()
    logits = torch.zeros(2, 124)
    logits[:, 0] = 2
    metric.update(logits, torch.tensor([0, 1]), 4.)
    counts = {i: 1 for i in range(124)}
    counts[1] = 500
    r = metric.compute(counts, [str(i) for i in range(124)])
    assert r["samples"] == 2 and r["supported_classes"] == 2 and r["nll"] == 2
    assert r["per_class"][0]["f1"] == pytest.approx(2/3)
    assert r["buckets"]["1-19"]["macro_f1_supported"] == pytest.approx(2/3)
    assert r["macro_f1_supported"] == pytest.approx(1/3)
    assert r["weighted_f1"] == pytest.approx(1/3)
    assert r["per_class"][123]["f1"] is None
    assert r["buckets"]["20-99"]["macro_f1_supported"] is None
    assert r["confusion"][1][0] == 1


def test_exact_ties_are_stable_across_chunks_and_do_not_change_logits():
    from src.e1hr.metrics import Metrics124, rank_logits
    x = torch.zeros(1, 124)
    original = x.clone()
    assert rank_logits(x)[0, :5].tolist() == [0, 1, 2, 3, 4]
    m = Metrics124()
    m.update(x, torch.tensor([4]), 1.)
    r = m.compute({i: 1 for i in range(124)}, [str(i) for i in range(124)])
    assert (r["top1"], r["top3"], r["top5"]) == (0., 0., 1.)
    assert torch.equal(original, x)
    x[:, 123] = 2
    m.update(x, torch.tensor([4]), 2.)
    r = m.compute({i: 1 for i in range(124)}, [str(i) for i in range(124)])
    assert r["per_class"][123]["predicted_count"] == 1


def test_invalid_and_empty_metrics_are_rejected():
    from src.e1hr.metrics import Metrics124
    m = Metrics124()
    with pytest.raises(ValueError):
        m.compute({}, [])
    with pytest.raises(ValueError):
        m.update(torch.full((1, 124), float("nan")), torch.tensor([0]), 0.)
    with pytest.raises(ValueError):
        m.update(torch.zeros(1, 124), torch.tensor([124]), 0.)


def test_fixed_validation_bucket_coverage_from_metadata_only():
    from src.e1hr.data import FixedManifest
    from src.e1hr.metrics import Metrics124
    manifest=FixedManifest()
    targets=torch.tensor([r.label_id for r in manifest.records('val')])
    metric=Metrics124(); metric.update(torch.zeros(len(targets),124),targets,7497.)
    result=metric.compute(manifest.n_train_by_label,manifest.labels)
    assert (result['output_classes'],result['supported_classes'],result['samples'])==(124,106,7497)
    for name,expected in {'1-19':(42,24,49),'20-99':(37,37,445),'100-499':(29,29,1513),'>=500':(16,16,5490),'1-2':(20,2,2)}.items():
        b=result['buckets'][name]
        assert (b['total_classes'],b['supported_classes'],b['samples'])==expected
