import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor

xp = backend.xp


class R2Score(BaseMetric):
    def __init__(self, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        preds = preds.data
        targets = targets.data

        SS_res = xp.sum((targets - preds)**2)
        SS_tot = xp.sum((targets - xp.mean(targets))**2)
        R2 = float(1 - SS_res / (SS_tot + self.eps))
        return R2