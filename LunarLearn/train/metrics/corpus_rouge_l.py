import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric, RougeL
from LunarLearn.core import Tensor

xp = backend.xp


class CorpusRougeL(BaseMetric):
    def __init__(self, beta: float = 1.2, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.eps = eps
        self._rouge_l_metric = RougeL(beta, eps)

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        scores = []
        for pred, tgt in zip(preds, targets):
            scores.append(self._rouge_l_metric(pred, tgt))

        return float(sum(scores) / (len(scores) + self.eps))