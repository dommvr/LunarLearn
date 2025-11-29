import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor
from LunarLearn.train.metrics import _lcs_length

xp = backend.xp


class RougeL(BaseMetric):
    def __init__(self, beta: float = 1.2, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.beta = beta
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        lcs = _lcs_length(preds, targets)

        m = len(preds)
        n = len(targets)

        if m == 0 or n == 0:
            return 0.0

        # LCS-based precision & recall
        R_lcs = lcs / (n + self.eps)
        P_lcs = lcs / (m + self.eps)

        # F-measure with β
        num = (1 + self.beta**2) * R_lcs * P_lcs
        den = R_lcs + self.beta**2 * P_lcs + self.eps

        return float(num / den)