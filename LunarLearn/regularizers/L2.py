import LunarLearn.backend as backend
from LunarLearn.tensor import ops
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer

xp = backend.xp

class L2(BaseRegularizer):
    """
    L2 Regularization (Weight Decay).
    Penalizes large weights by adding 0.5 * λ * ||W||².

    Args:
        lam (float): Regularization strength.
    """
    def __init__(self, lam=1e-4, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = xp.array(lam, dtype=backend.DTYPE)

    def loss(self, param):
        """Compute L2 regularization loss for a single parameter tensor."""
        return 0.5 * self.lam * ops.sum(param ** 2)