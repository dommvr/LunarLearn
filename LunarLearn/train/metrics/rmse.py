from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor, ops


class RMSE(BaseMetric):
    def __init__(self):
        super().__init__(expect_vector=False)

    def compute(self, preds: Tensor, targets: Tensor, **kwargs):
        return ops.sqrt(ops.mean_squared_error(preds, targets))