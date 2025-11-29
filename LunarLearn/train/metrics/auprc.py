import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _binary_auprc

xp = backend.xp

class Auprc(BaseMetric):
    def __init__(self, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.eps = eps

    def compute(self, preds, targets, **kwargs):
        preds = preds.data
        targets = targets.data

        # MULTI-LABEL or BINARY (N, C)
        if preds.ndim == 2:
            C = preds.shape[1]
            per_class_auprc = xp.zeros(C)

            for c in range(C):
                per_class_auprc[c] = _binary_auprc(preds[:, c], targets[:, c], self.eps)

            # micro AUPRC: flatten everything (sklearn style)
            micro_auprc = _binary_auprc(preds.ravel(), targets.ravel(), self.eps)

            # macro AUPRC: unweighted mean
            macro_auprc = float(xp.mean(per_class_auprc))

            # weighted AUPRC: weight by class support
            # support = number of positive samples for each class
            support = xp.sum(targets == 1, axis=0)
            total_support = xp.sum(support) + self.eps
            weighted_auprc = float(xp.sum(per_class_auprc * support) / total_support)

            return micro_auprc, macro_auprc, weighted_auprc, per_class_auprc

        # MULTI-CLASS (N,)
        elif preds.ndim == 1:
            raise ValueError("AUPRC requires probabilities per class, not class indices.")

        else:
            raise ValueError("Unsupported shape for preds or targets.")