import LunarLearn.backend as backend
from LunarLearn.tensor import ops
from LunarLearn.tensor import Tensor
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer

xp = backend.xp

class SpectralNorm(BaseRegularizer):
    """
    Spectral Norm Regularizer.
    Penalizes the largest singular value (o_max) of a parameter.

    Args:
        lam (float): Regularization strength.
        n_iter (int): Power iteration steps for approximation.
    """
    def __init__(self, lam=1e-4, n_iter=1, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = backend.xp.array(lam, dtype=backend.DTYPE)
        self.n_iter = n_iter

    def _spectral_norm(self, param):
        """Approximate o_max(W) via power iteration."""
        W_flat = ops.reshape(param, (param.shape[0], -1))
        u = Tensor.randn((W_flat.shape[0], 1))
        for _ in range(self.n_iter):
            v = ops.normalize(ops.matmul(ops.transpose(W_flat), u))
            u = ops.normalize(ops.matmul(W_flat, v))
        sigma = ops.matmul(ops.transpose(u), ops.matmul(W_flat, v))
        return ops.squeeze(sigma)

    def loss(self, param):
        """Compute spectral norm penalty: λ * o_max(W)."""
        return self.lam * self._spectral_norm(param)