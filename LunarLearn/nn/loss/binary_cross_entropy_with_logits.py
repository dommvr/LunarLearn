import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

DTYPE = backend.DTYPE

class BinaryCrossEntropyWithLogits(BaseLoss):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.binary_cross_entropy_with_logits(predictions, targets)