import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class KLDivergence(BaseLoss):
    """
    Kullback-Leibler (KL) Divergence loss with autograd support.

    This loss measures the divergence between two probability distributions. 
    It is often used when comparing a predicted probability distribution to 
    a target distribution (e.g., soft labels or teacher-student models).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the KL divergence over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted probabilities (softmax output) of shape (B, C).
                targets (Tensor): Target probability distributions of shape (B, C).
                epsilon (float, optional): Small constant to avoid log(0). Default is 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean KL divergence. Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
        return ops.kl_divergence(predictions, targets, epsilon=epsilon)
