"""Full 124-output ranking; macro-F1 averages ground-truth-supported classes."""
import torch


class ClassificationMetrics:
    def __init__(self, classes=124):
        self.classes = classes
        self.confusion = torch.zeros(classes, classes, dtype=torch.int64)
        self.correct = {k: 0 for k in (1, 3, 5)}
        self.n = 0
        self.loss_sum = 0.

    def update(self, logits, targets, loss_sum=0.):
        logits, targets = logits.detach().float().cpu(), targets.detach().long().cpu()
        if logits.shape != (len(targets), self.classes) or not torch.isfinite(logits).all():
            raise ValueError("Invalid/nonfinite logits")
        if targets.numel() and (targets.min() < 0 or targets.max() >= self.classes):
            raise ValueError("Invalid target ID")
        ranking = logits.topk(5, dim=1).indices
        for k in self.correct:
            self.correct[k] += (ranking[:, :k] == targets[:, None]).any(1).sum().item()
        self.confusion += torch.bincount(targets * self.classes + ranking[:, 0],
                                        minlength=self.classes**2).reshape(self.classes, self.classes)
        self.n += len(targets)
        self.loss_sum += float(loss_sum)

    def compute(self):
        if not self.n:
            raise ValueError("Cannot compute metrics without evaluation samples")
        matrix = self.confusion.double()
        support, predicted, tp = matrix.sum(1), matrix.sum(0), matrix.diag()
        precision = tp / predicted.clamp_min(1)
        recall = tp / support.clamp_min(1)
        f1 = 2 * tp / (support + predicted).clamp_min(1)
        mask = support > 0
        return {**{f"top{k}": self.correct[k] / self.n for k in self.correct},
                "macro_f1_supported": f1[mask].mean().item(),
                "weighted_f1": (f1 * support).sum().item() / self.n,
                "nll": self.loss_sum / self.n, "samples": self.n,
                "output_classes": self.classes, "supported_classes": int(mask.sum()),
                "unsupported_classes": (~mask).nonzero().flatten().tolist(),
                "per_class": [{"label_id": i, "precision": precision[i].item(),
                               "recall": recall[i].item(), "f1": f1[i].item() if mask[i] else None,
                               "support": int(support[i]), "predicted_count": int(predicted[i])}
                              for i in range(self.classes)]}
