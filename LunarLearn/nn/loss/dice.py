from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor

class Dice(BaseLoss):
    def forward(self, predictions: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
        probs = predictions.reshape(-1)
        targets = targets.reshape(-1)
        inter = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2.0 * inter + smooth) / (union + smooth)
        loss = 1.0 - dice
        loss.grad_fn = "dice"
        return loss