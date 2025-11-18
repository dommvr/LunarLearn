import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class Focal(BaseLoss):
    """
    Focal loss with autograd support.

    This loss is designed to address class imbalance by down-weighting well-classified examples.
    It extends the standard cross-entropy loss by a modulating factor that reduces the loss for
    easy examples.

    Args:
        alpha (float, optional): Weighting factor for the rare class. Default is 1.0.
        gamma (float, optional): Focusing parameter that reduces loss contribution from easy examples. Default is 2.0.

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the focal loss over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted probabilities (softmax output) of shape (B, C).
                targets (Tensor): Target labels. Either integer indices (B,) or one-hot (B, C).
                epsilon (float, optional): Small constant to avoid log(0). Default is 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean focal loss. Gradients are tracked automatically.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__(trainable=False)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.focal(predictions, targets, alpha=self.alpha, gamma=self.gamma)