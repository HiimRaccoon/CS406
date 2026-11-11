import math
import pytest
import torch


def test_ties_supported_f1_cross_bucket_false_positive_and_nll():
    from src.e3.metrics import Metrics124, rank_logits
    logits = torch.zeros(2,124)
    assert rank_logits(logits)[0,:5].tolist() == [0,1,2,3,4]
    m = Metrics124(); m.update(logits,torch.tensor([0,1]),2*math.log(124))
    support = {i:20 for i in range(124)}; support[0]=1; support[1]=500
    result = m.compute(support,[str(i) for i in range(124)])
    assert result["per_class"][0]["f1"] == pytest.approx(2/3)
    assert result["macro_f1_supported"] == pytest.approx(1/3)
    assert result["buckets"]["1-19"]["macro_f1_supported"] == pytest.approx(2/3)
    assert result["per_class"][2]["f1"] is None
    assert result["per_class"][2]["precision"] == result["per_class"][2]["recall"] == 0
    assert result["buckets"]["100-499"]["top1"] is None
    assert result["nll"] == pytest.approx(math.log(124))


def test_fixed_bucket_coverage_metadata_only():
    from src.e3.metrics import bucket_membership
    from src.e3.data import FixedManifest
    m = FixedManifest()
    buckets = bucket_membership(m.n_train)
    assert {k:len(v) for k,v in buckets.items()} == {"1-19":42,"20-99":37,"100-499":29,">=500":16,"1-2":20}
    assert sum(row["label_id"] in buckets["1-2"] for row in m.records("val")) == 2
