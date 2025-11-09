import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

DTYPE = backend.DTYPE

class BinaryCrossEntropyDice(BaseLoss):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        bce_loss = ops.binary_cross_entropy(predictions, targets)
        probs = ops.sigmoid(predictions)
        dice_loss = ops.dice(probs, targets)
        loss = bce_loss + dice_loss
        loss.grad_fn = "binary_cross_entropy_dice"
        return loss