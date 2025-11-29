import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import true_positive, false_negative
from LunarLearn.core import Tensor

xp = backend.xp

class Recall(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.threshold = threshold
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        # true positives and false negative
        total_tp, per_class_tp = true_positive(preds, targets, self.threshold)
        total_fn, per_class_fn = false_negative(preds, targets, self.threshold)

        # avoid division by zero
        per_class_recall = per_class_tp / (per_class_tp + per_class_fn + self.eps)

        # micro recall: TP / (TP + FN)
        micro_recall = float(total_tp / (total_tp + total_fn + self.eps))

        # macro recall: mean over classes
        macro_recall = float(xp.mean(per_class_recall))

        # weighted recall: weight by support (true occurrences of each class)
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
        weighted_recall = float(xp.sum(per_class_recall * support) / total_support)

        return micro_recall, macro_recall, weighted_recall, per_class_recall