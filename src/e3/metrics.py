"""Unchanged E1 metric/ranking semantics and fixed train-frequency membership."""
from ..e1.metrics import Metrics124, rank_logits


def bucket_membership(n_train: dict) -> dict:
    if set(n_train) != set(range(124)) or any(type(n) is not int or n < 1 for n in n_train.values()):
        raise ValueError("All 124 positive unique train supports are required")
    return {name: tuple(c for c in range(124) if low <= n_train[c] <= high)
            for name, low, high in [("1-19",1,19),("20-99",20,99),("100-499",100,499),(">=500",500,float("inf")),("1-2",1,2)]}


__all__ = ["Metrics124", "rank_logits", "bucket_membership"]
