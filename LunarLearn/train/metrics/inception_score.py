import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor, ops

xp = backend.xp


class InceptionScore(BaseMetric):
    def __init__(self, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.eps = eps

    def compute(self, preds: Tensor):
        # compute p(y|x)
        # use your ops.log_softmax since it's stable
        log_probs = ops.log_softmax(preds, axis=1, epsilon=self.eps).data
        probs = xp.exp(log_probs)  # shape (N, C)

        # p(y) = mean over samples
        py = xp.mean(probs, axis=0)  # shape (C,)

        # KL divergence per sample:
        # sum_c  p(y|x) * (log p(y|x) - log p(y))
        log_py = xp.log(py + self.eps)
        kl_per_sample = xp.sum(probs * (log_probs - log_py[None, :]), axis=1)

        # average KL
        kl_mean = xp.mean(kl_per_sample)

        # IS = exp(expected KL)
        return float(xp.exp(kl_mean))