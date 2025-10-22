import LunarLearn.backend as backend
from LunarLearn.tensor import ops
from LunarLearn.tensor import Tensor
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer

xp = backend.xp

class NuclearNorm(BaseRegularizer):
    """
    Nuclear Norm Regularizer.
    Encourages low-rank weights by penalizing the sum of singular values.

    Args:
        lam (float): Regularization strength.
    """
    def __init__(self, lam=1e-4, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = xp.array(lam, dtype=backend.DTYPE)

    def loss(self, param):
        """Compute nuclear norm penalty: λ * Σ o_i(W)."""
        W_flat = ops.reshape(param, (param.shape[0], -1))
        _, s, _ = ops.svd(W_flat)
        return self.lam * ops.sum(s)