from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor, ops


class MAE(BaseMetric):
    def __init__(self):
        super().__init__(expect_vector=False)

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        return ops.mean_absolute_error(preds, targets)