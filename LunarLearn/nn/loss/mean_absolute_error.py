import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class MeanAbsoluteError(BaseLoss):
    """
    Mean Absolute Error (MAE) loss with autograd support.

    This loss computes the average absolute difference between predictions and targets.
    It supports integer class labels (converted to one-hot) or direct target values.

    Methods:
        forward(predictions: Tensor, targets: Tensor) -> Tensor:
            Computes the MAE over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted values of shape (B, C) or (B, 1).
                targets (Tensor): Target values. Either integer class indices (B,) 
                                  or one-hot / continuous targets (B, C).

            Returns:
                Tensor: Scalar tensor containing the mean absolute error. 
                        Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Convert integer labels to one-hot if needed
        if targets.ndim == 1:
            targets_int = targets.astype(int)
            targets_onehot = ops.eye(predictions.shape[1], dtype=DTYPE)[targets_int]
        else:
            targets_onehot = Tensor(targets, requires_grad=False)

        # Clip targets for numerical stability
        targets_onehot = ops.clip(targets_onehot, 0.0, 1.0)

        # MAE elementwise
        loss_tensor = ops.abs(predictions - targets_onehot)
        loss = ops.mean(loss_tensor)
        loss.grad_fn = "mae"
        return loss
