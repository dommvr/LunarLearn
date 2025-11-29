import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric, RougeL
from LunarLearn.core import Tensor
from typing import List

xp = backend.xp


class CorpusRougeLMultiRef(BaseMetric):
    def __init__(self, beta: float = 1.2, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.eps = eps
        self._rouge_l_metric = RougeL(beta, eps)

    def compute(self, preds: List[Tensor], list_of_tgt_lists: List[List[Tensor]], **kwargs):
        scores = []
        for cand, refs in zip(preds, list_of_tgt_lists):
            best = 0.0
            for ref in refs:
                score = self._rouge_l_metric(cand, ref)
                if score > best:
                    best = score
            scores.append(best)

        return float(sum(scores) / (len(scores) + self.eps))