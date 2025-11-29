import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _activation_stats, _matrix_sqrt
from LunarLearn.core import Tensor

xp = backend.xp


class FID(BaseMetric):
    def __init__(self, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor):
        mu_r, cov_r = _activation_stats(preds, self.eps)
        mu_g, cov_g = _activation_stats(targets, self.eps)

        # mean difference term
        diff = mu_r - mu_g
        diff_sq = float(diff @ diff)

        # covariance term
        cov_prod = cov_r @ cov_g
        cov_prod_sqrt = _matrix_sqrt(cov_prod + self.eps * xp.eye(cov_r.shape[0], dtype=cov_r.dtype), eps=self.eps)

        trace_r = float(xp.trace(cov_r))
        trace_g = float(xp.trace(cov_g))
        trace_sqrt = float(xp.trace(cov_prod_sqrt))

        fid_value = diff_sq + trace_r + trace_g - 2.0 * trace_sqrt

        # numerical noise can make this slightly negative
        if fid_value < 0:
            fid_value = 0.0

        return float(fid_value)