import LunarLearn.backend as backend
from LunarLearn.tensor import ops
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer

xp = backend.xp

class Orthogonal(BaseRegularizer):
    """
    Orthogonal Regularization.
    Loss = λ * ||WᵀW - I||_F²
    Encourages orthogonality in weights.
    """

    def __init__(self, lam=1e-4, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = xp.array(lam, dtype=backend.DTYPE)

    def loss(self, param):
        """Compute orthogonality penalty."""
        W_flat = ops.reshape(param, (param.shape[0], -1))
        WT_W = ops.matmul(ops.transpose(W_flat), W_flat)
        I = ops.eye(WT_W.shape[0])
        return self.lam * ops.sum((WT_W - I) ** 2)