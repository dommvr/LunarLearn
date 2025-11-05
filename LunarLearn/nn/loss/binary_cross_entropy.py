import LunarLearn.backend as backend
from LunarLearn.loss.BaseLoss import BaseLoss
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE

class BinaryCrossEntropyLoss(BaseLoss):
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
        # Convert integer labels to one-hot if necessary
        if targets.ndim == 1:
            targets_int = targets.astype(int)
            targets_onehot = ops.eye(predictions.shape[1], dtype=DTYPE)[targets_int]
        else:
            targets_onehot = Tensor(targets, requires_grad=False)

        # Clip predictions to avoid log(0)
        preds_clipped = ops.clip(predictions, epsilon, 1 - epsilon)

        # Compute BCE loss elementwise
        loss_tensor = -(targets_onehot * ops.log(preds_clipped) +
                        (1 - targets_onehot) * ops.log(1 - preds_clipped))

        # Take mean over batch
        loss = ops.mean(loss_tensor)
        loss.grad_fn = "binary_cross_entropy"

        return loss