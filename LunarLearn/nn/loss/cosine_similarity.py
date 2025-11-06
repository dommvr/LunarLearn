import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class CosineSimilarity(BaseLoss):
    """
    Cosine similarity loss with autograd support.

    Computes 1 - cosine similarity between predicted vectors and target vectors.
    Fully compatible with autograd: gradients flow through normalization and
    vector operations.

    Args:
        None (all parameters are passed to `forward`).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the mean 1 - cosine similarity loss over a batch.

            Args:
                predictions (Tensor): Predicted vectors of shape (B, D).
                targets (Tensor): Target vectors of shape (B, D).
                epsilon (float, optional): Small constant to avoid division by zero. Default 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean loss. Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
        predictions = predictions.astype(DTYPE)
        targets = targets.astype(DTYPE)

        # Normalize vectors
        pred_norm = ops.norm(predictions, axis=1, keepdims=True) + epsilon
        targ_norm = ops.norm(targets, axis=1, keepdims=True) + epsilon

        # Cosine similarity
        cosine_sim = ops.sum(predictions * targets, axis=1) / (pred_norm * targ_norm)
        loss = ops.mean(1.0 - cosine_sim)
        loss.grad_fn = "cosine_similarity"

        return loss
