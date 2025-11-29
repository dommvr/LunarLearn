import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor

xp = backend.xp

class TopKAccuracy(BaseMetric):
    def __init__(self, k: int = 5):
        super().__init__(expect_vector=False)
        self.k = k

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        preds = preds.data
        targets = targets.data

        if preds.ndim != 2:
            raise ValueError("topk_accuracy requires prediction matrix of shape (N, C).")

        # sort descending, take top-k indices
        topk = xp.argsort(-preds, axis=1)[:, :self.k]

        # check whether true label is in the row's top-k
        correct = xp.any(topk == targets[:, None], axis=1)

        return float(xp.mean(correct))