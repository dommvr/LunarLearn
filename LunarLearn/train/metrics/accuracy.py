import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor

xp = backend.xp

class Accuracy(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.threshold = threshold
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        preds = preds.data
        targets = targets.data

        # MULTI-LABEL or BINARY (N, C)
        if preds.ndim == 2:
            # threshold if necessary
            if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
                preds_bin = (preds > self.threshold).astype(int)
            else:
                preds_bin = preds

            correct = (preds_bin == targets)

            # per-class accuracy
            per_class_accuracy = xp.mean(correct, axis=0)

            # micro accuracy (flatten everything)
            micro_accuracy = float(xp.mean(correct))

            # macro accuracy (mean over classes)
            macro_accuracy = float(xp.mean(per_class_accuracy))

            # weighted accuracy (weight by support)
            support = xp.sum(targets == 1, axis=0)
            total_support = xp.sum(support) + self.eps
            weighted_accuracy = float(xp.sum(per_class_accuracy * support) / total_support)

            return micro_accuracy, macro_accuracy, weighted_accuracy, per_class_accuracy

        # MULTI-CLASS or BINARY (N,)
        elif preds.ndim == 1:
            # per-sample correctness
            correct = preds == targets
            micro_accuracy = float(xp.mean(correct))

            # per-class accuracy
            C = int(xp.max(preds.max(), targets.max()) + 1)
            per_class_accuracy = xp.zeros(C)

            for c in range(C):
                mask = (targets == c)
                if xp.sum(mask) == 0:
                    per_class_accuracy[c] = 0.0
                else:
                    per_class_accuracy[c] = float(xp.mean(correct[mask]))

            macro_accuracy = float(xp.mean(per_class_accuracy))

            # weighted by class frequency
            support = xp.array([xp.sum(targets == c) for c in range(C)], dtype=float)
            total_support = xp.sum(support) + self.eps
            weighted_accuracy = float(xp.sum(per_class_accuracy * support) / total_support)

            return micro_accuracy, macro_accuracy, weighted_accuracy, per_class_accuracy

        else:
            raise ValueError("Unsupported shape for preds or targets.")