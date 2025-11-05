import LunarLearn.backend as backend
from LunarLearn.loss.BaseLoss import BaseLoss
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE

class HuberLoss(BaseLoss):
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

        delta = Tensor(delta, requires_grad=False, dtype=predictions.dtype)

        # --- Handle targets for classification ---
        if targets.ndim == 1 and predictions.ndim == 2 and predictions.shape[1] > 1:
            # integer labels -> one-hot
            targets_int = targets.astype(int)
            targets = ops.eye(predictions.shape[1])[targets_int]
        elif targets.ndim == 2 and predictions.ndim == 2 and targets.shape[1] == predictions.shape[1]:
            targets = Tensor(targets, requires_grad=False)
        elif targets.ndim == 1 and predictions.ndim == 2 and predictions.shape[1] == 1:
            targets = Tensor(targets.reshape(-1, 1), requires_grad=False)
        elif targets.ndim == 2 and predictions.ndim == 2 and predictions.shape[1] == 1:
            targets = Tensor(targets, requires_grad=False)
        # else regression mode

        # --- Compute error ---
        error = predictions - targets
        abs_error = ops.abs(error)

        quadratic = 0.5 * (error * error)
        linear = delta * (abs_error - 0.5 * delta)

        loss_tensor = ops.where(abs_error <= delta, quadratic, linear)
        loss = ops.mean(loss_tensor)
        loss.grad_fn = "huber_loss"
        return loss