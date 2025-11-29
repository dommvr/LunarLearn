import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import true_positive, false_positive
from LunarLearn.core import Tensor

xp = backend.xp

class Precision(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.threshold = threshold
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        # true positives and false positives
        total_tp, per_class_tp = true_positive(preds, targets, self.threshold)
        total_fp, per_class_fp = false_positive(preds, targets, self.threshold)

        # avoid division by zero
        per_class_precision = per_class_tp / (per_class_tp + per_class_fp + self.eps)

        # micro precision: TP / (TP + FP)
        micro_precision = float(total_tp / (total_tp + total_fp + self.eps))

        # macro precision: mean over classes
        macro_precision = float(xp.mean(per_class_precision))

        # weighted precision: weight by support (true occurrences of each class)
        # Works for multi-class and multi-label.
        # For multi-label: support = number of positives for that label.
        # For multi-class: support = count of samples belonging to class c.
        targets_data = targets.data
        if targets_data.ndim == 2:
            support = xp.sum(targets_data == 1, axis=0)
        else:
            C = len(per_class_tp)
            support = xp.array([xp.sum(targets_data == c) for c in range(C)])

        total_support = xp.sum(support) + self.eps
        weighted_precision = float(xp.sum(per_class_precision * support) / total_support)

        return micro_precision, macro_precision, weighted_precision, per_class_precision