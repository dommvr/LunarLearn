import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class BinaryCrossEntropy(BaseLoss):
    """
    Binary Cross-Entropy (BCE) loss with autograd support.

    This loss computes the binary cross-entropy between predicted probabilities
    and target labels. It supports one-hot or integer-encoded targets, is fully
    compatible with the autograd system, and clips predictions to avoid log(0).

    Args:
        None (all parameters are passed to `forward`).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the BCE loss over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted probabilities of shape (m, n_classes) in [0, 1].
                targets (Tensor): True labels. Either integer-encoded (shape (m,))
                                  or one-hot (shape (m, n_classes)).
                epsilon (float, optional): Small value to clip predictions. Default is 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean BCE loss over the batch.
                        Gradients are tracked automatically for autograd.
    """
    def forward(self, predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
        return ops.binary_cross_entropy(predictions, targets, epsilon=epsilon)