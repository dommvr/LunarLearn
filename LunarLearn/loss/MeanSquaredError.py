import LunarLearn.backend as backend
from LunarLearn.loss.BaseLoss import BaseLoss
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE

class MeanSquaredErrorLoss(BaseLoss):
    """
    Mean Squared Error (MSE) loss with autograd support.

    This loss computes the average squared difference between predictions and targets.
    It supports integer class labels (converted to one-hot) or direct target values.

    Methods:
        forward(predictions: Tensor, targets: Tensor) -> Tensor:
            Computes the MSE over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted values of shape (B, C) or (B, 1).
                targets (Tensor): Target values. Either integer class indices (B,) 
                                or one-hot / continuous targets (B, C).

            Returns:
                Tensor: Scalar tensor containing the mean squared error. 
                        Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor, epsilon: float = 1e-7) -> Tensor:
        # Convert integer labels to one-hot if necessary
        if targets.ndim == 1:
            targets_int = targets.astype(int)
            targets_onehot = ops.eye(predictions.shape[1], dtype=DTYPE)[targets_int]
        else:
            targets_onehot = Tensor(targets, requires_grad=False)

        # Clip for numerical stability
        targets_onehot = ops.clip(targets_onehot, 0.0, 1.0)

        diff = predictions - targets_onehot
        loss = ops.mean(diff * diff)
        loss.grad_fn = "mean_squared_error"

        return loss
