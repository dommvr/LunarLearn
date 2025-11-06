import LunarLearn.backend as backend
from LunarLearn.loss.BaseLoss import BaseLoss
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE

class CategoricalCrossEntropy(BaseLoss):
    """
    Categorical cross-entropy loss with autograd support.

    Computes the mean cross-entropy between predicted probabilities and target labels.
    Fully compatible with autograd: gradients flow through log, multiplication, and mean.

    Args:
        None (all parameters are passed to `forward`).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the mean categorical cross-entropy loss.

            Args:
                predictions (Tensor): Predicted probabilities (softmax output) of shape (B, C).
                targets (Tensor): Target labels. Either integer indices (B,) or one-hot (B, C).
                epsilon (float, optional): Small constant to avoid log(0). Default 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean loss. Gradients are tracked automatically.
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

        # Compute cross-entropy elementwise
        loss_tensor = -targets_onehot * ops.log(preds_clipped)

        # Mean over batch
        loss = ops.mean(ops.sum(loss_tensor, axis=1))
        loss.grad_fn = "categorical_cross_entropy"

        return loss