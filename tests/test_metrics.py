import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from src.pilot.metrics import ClassificationMetrics


def test_supported_macro_keeps_cross_class_false_positives():
    rng = np.random.default_rng(14)
    logits = torch.tensor(rng.normal(size=(12, 124)), dtype=torch.float32)
    y = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 1, 2])
    # Predictions include unsupported class 123 and mistakes across supported classes.
    pred = np.array([0, 1, 123, 1, 0, 2, 0, 2, 123, 0, 1, 2])
    logits[torch.arange(12), torch.from_numpy(pred)] = 20
    m = ClassificationMetrics()
    m.update(logits[:5], y[:5], 5.)
    m.update(logits[5:], y[5:], 7.)
    result = m.compute()
    assert result["macro_f1_supported"] == pytest.approx(f1_score(y, pred, labels=[0, 1, 2], average="macro"))
    assert result["weighted_f1"] == pytest.approx(f1_score(y, pred, average="weighted"))
    assert result["top1"] == pytest.approx((pred == y.numpy()).mean())
    assert result["top1"] <= result["top3"] <= result["top5"]
    assert result["per_class"][123]["f1"] is None
    assert result["per_class"][123]["predicted_count"] == 2
    assert result["supported_classes"] == 3
    assert result["nll"] == 1
