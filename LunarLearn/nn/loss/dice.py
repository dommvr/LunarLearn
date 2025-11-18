from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

class Dice(BaseLoss):
    def forward(self, predictions: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
        return ops.dice(predictions, targets, smooth=smooth)