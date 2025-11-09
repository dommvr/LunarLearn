import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

DTYPE = backend.DTYPE

class BinaryCrossEntropyWithLogits(BaseLoss):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = ops.mean(
            -targets * ops.log_sigmoid(predictions) -
            (1 - targets) * ops.log_sigmoid(-predictions)
        )
        loss.grad_fn = "binary_cross_entropy_with_logits"
        return loss