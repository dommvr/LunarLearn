import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric, Precision, Recall
from LunarLearn.core import Tensor

xp = backend.xp

class F1_score(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.threshold = threshold
        self.eps = eps
        self._precision_metric = Precision(threshold, eps)
        self._recall_metric = Recall(threshold, eps)

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        micro_precision, macro_precision, weighted_precision, per_class_precision = self._precision_metric(preds, targets)
        micro_recall, macro_recall, weighted_recall, per_class_recall = self._recall_metric(preds, targets)

        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + self.eps)
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + self.eps)
        weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall + self.eps)
        per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + self.eps)

        return micro_f1, macro_f1, weighted_f1, per_class_f1