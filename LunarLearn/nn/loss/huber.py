import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class Huber(BaseLoss):
    """
    Huber loss with autograd support.

    The Huber loss is less sensitive to outliers than mean squared error (MSE). 
    It behaves like MSE for small errors and like mean absolute error (MAE) for 
    large errors, with a transition defined by the delta threshold.

    Args:
        delta (float, optional): Threshold at which the loss switches from quadratic 
            to linear. Default is 1.0.

    Methods:
        forward(predictions: Tensor, targets: Tensor, delta: float = 1.0) -> Tensor:
            Computes the Huber loss over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted values of shape (B, D) or (B, C).
                targets (Tensor): Target values or labels. Can be integer indices, one-hot,
                    or continuous values depending on the task.
                delta (float, optional): Threshold at which to switch from quadratic to linear loss.
                    Default is 1.0.

            Returns:
                Tensor: Scalar tensor containing the mean Huber loss. Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor, delta: float = 1.0) -> Tensor:
        return ops.huber(predictions, targets, delta=delta)