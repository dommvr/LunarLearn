import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _binary_auroc

xp = backend.xp

class Auroc(BaseMetric):
    def __init__(self, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.eps = eps

    def compute(self, preds, targets, **kwargs):
        preds = preds.data
        targets = targets.data

        # MULTI-LABEL / BINARY in (N, C) form
        if preds.ndim == 2:
            C = preds.shape[1]
            per_class_auroc = xp.zeros(C, dtype=xp.float32)

            for c in range(C):
                per_class_auroc[c] = _binary_auroc(preds[:, c], targets[:, c], self.eps)

            # micro AUROC: flatten all scores and labels
            micro_auroc = _binary_auroc(preds.ravel(), targets.ravel(), self.eps)

            # macro AUROC: unweighted mean over classes
            macro_auroc = float(xp.mean(per_class_auroc))

            # weighted AUROC: weight by number of positives per class
            support = xp.sum(targets == 1, axis=0)          # shape (C,)
            total_support = xp.sum(support) + self.eps
            weighted_auroc = float(xp.sum(per_class_auroc * support) / total_support)

            return micro_auroc, macro_auroc, weighted_auroc, per_class_auroc

        # preds is (N,) → you gave class indices, not scores
        elif preds.ndim == 1:
            raise ValueError("AUROC requires probability scores per class, not class indices.")

        else:
            raise ValueError("Unsupported shape for preds or targets.")