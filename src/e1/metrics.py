"""Full-split 124-output metrics with deterministic ranking and fixed buckets."""
import math

import torch


def rank_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 124 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite [N,124] logits")
    # Stable sorting preserves the input label-index order for exact ties.
    return torch.argsort(logits, dim=1, descending=True, stable=True)


class Metrics124:
    def __init__(self):
        self.confusion = torch.zeros(124, 124, dtype=torch.int64)
        self.correct_by_label = {k: torch.zeros(124, dtype=torch.int64) for k in (1, 3, 5)}
        self.n, self.loss_sum = 0, 0.

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_sum: float) -> None:
        logits, targets = logits.detach().cpu(), targets.detach().cpu()
        if (targets.ndim != 1 or targets.dtype not in {torch.int64, torch.int32}
                or len(targets) != len(logits) or (targets.numel() and (targets.min() < 0 or targets.max() >= 124))
                or not math.isfinite(float(loss_sum)) or loss_sum < 0):
            raise ValueError("Invalid metric targets or loss sum")
        ranking = rank_logits(logits)
        for k in self.correct_by_label:
            hits = (ranking[:, :k] == targets[:, None]).any(1)
            self.correct_by_label[k] += torch.bincount(targets[hits].long(), minlength=124)
        self.confusion += torch.bincount(targets.long() * 124 + ranking[:, 0], minlength=124**2).reshape(124, 124)
        self.n += len(targets)
        self.loss_sum += float(loss_sum)

    def compute(self, n_train: dict, labels: list[str]) -> dict:
        if (not self.n or len(labels) != 124 or set(n_train) != set(range(124))
                or any(type(v) is not int or v < 1 for v in n_train.values())):
            raise ValueError("Nonempty evaluation and complete fixed label/train-support metadata required")
        matrix = self.confusion.double()
        support, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diag()
        precision = tp / predicted.clamp_min(1)
        recall = tp / support.clamp_min(1)
        f1 = 2 * tp / (support + predicted).clamp_min(1)
        evaluated = support > 0
        def bucket_name(n):
            return "1-19" if n < 20 else "20-99" if n < 100 else "100-499" if n < 500 else ">=500"
        per_class = [{"label_id": i, "label": labels[i], "n_train": n_train[i], "bucket": bucket_name(n_train[i]),
                      "support": int(support[i]), "predicted_count": int(predicted[i]),
                      "precision": precision[i].item(), "recall": recall[i].item(),
                      "f1": f1[i].item() if evaluated[i] else None} for i in range(124)]
        buckets = {}
        for name, lo, hi in [("1-19", 1, 19), ("20-99", 20, 99), ("100-499", 100, 499),
                             (">=500", 500, math.inf), ("1-2", 1, 2)]:
            members = torch.tensor([lo <= n_train[i] <= hi for i in range(124)])
            supported = members & evaluated
            samples = int(support[members].sum())
            buckets[name] = {"total_classes": int(members.sum()), "supported_classes": int(supported.sum()),
                             "samples": samples, "macro_f1_supported": f1[supported].mean().item() if samples else None,
                             **{f"top{k}": int(hits[members].sum()) / samples if samples else None
                                for k, hits in self.correct_by_label.items()}}
        return {"samples": self.n, "output_classes": 124, "supported_classes": int(evaluated.sum()),
                "unsupported_classes": (~evaluated).nonzero().flatten().tolist(), "nll": self.loss_sum / self.n,
                "macro_f1_supported": f1[evaluated].mean().item(), "weighted_f1": (f1 * support).sum().item() / self.n,
                **{f"top{k}": int(hits.sum()) / self.n for k, hits in self.correct_by_label.items()},
                "per_class": per_class, "buckets": buckets, "confusion": self.confusion.tolist()}
