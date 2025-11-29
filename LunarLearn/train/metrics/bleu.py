import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor
from LunarLearn.train.metrics import _modified_precision

xp = backend.xp


class BLEU(BaseMetric):
    def __init__(self, max_n: int = 4, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.max_n = max_n
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        # Modified precisions p1..pN
        precisions = []
        for n in range(1, self.max_n + 1):
            precisions.append(_modified_precision(preds, targets, n))

        # If any precision is zero, BLEU would collapse to zero.
        # We epsilon it to avoid log(0).
        precisions = [p if p > 0 else self.eps for p in precisions]

        # Geometric mean of the precisions
        log_sum = sum((1 / self.max_n) * xp.log(p) for p in precisions)
        geo_mean = xp.exp(log_sum)

        # Brevity penalty
        c = len(preds)
        r = len(targets)

        if c == 0:
            return 0.0

        if c < r:
            BP = xp.exp(1 - r / (c + self.eps))
        else:
            BP = 1.0

        return float(BP * geo_mean)