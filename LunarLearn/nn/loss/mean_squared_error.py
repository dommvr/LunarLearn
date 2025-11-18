import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class MeanSquaredError(BaseLoss):
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
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.mean_squared_error(predictions, targets)
