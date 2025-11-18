import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class Triplet(BaseLoss):
    """
    Triplet loss with autograd support.

    This loss encourages the distance between an anchor and a positive sample 
    (same class) to be smaller than the distance between the anchor and a negative 
    sample (different class) by at least a margin. It is commonly used in 
    metric learning and embedding models.

    Methods:
        forward(anchor: Tensor, positive: Tensor, negative: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the triplet loss over a batch of embeddings.

            Args:
                anchor (Tensor): Anchor embeddings of shape (B, D).
                positive (Tensor): Positive embeddings (same class as anchor), shape (B, D).
                negative (Tensor): Negative embeddings (different class), shape (B, D).
                epsilon (float, optional): Small constant for numerical stability. 
                                        Default is 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean triplet loss. 
                        Gradients are tracked automatically.
    """
    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 0.2, epsilon: float = 1e-15) -> Tensor:
        return ops.triplet(anchor, positive, negative, margin=margin, epsilon=epsilon)