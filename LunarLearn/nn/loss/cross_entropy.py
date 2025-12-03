import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class CrossEntropy(BaseLoss):
    """
    Cross-entropy loss with autograd support.

    Computes the mean cross-entropy between predicted probabilities and target labels.
    Fully compatible with autograd: gradients flow through log, multiplication, and mean.

    Args:
        None (all parameters are passed to `forward`).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the mean cross-entropy loss.

            Args:
                predictions (Tensor): Predicted probabilities (softmax output) of shape (B, C).
                targets (Tensor): Target labels. Either integer indices (B,) or one-hot (B, C).
                epsilon (float, optional): Small constant to avoid log(0). Default 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean loss. Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor, axis: int = -1, epsilon: float = 1e-15) -> Tensor:
        return ops.cross_entropy(predictions, targets, axis=axis, epsilon=epsilon)