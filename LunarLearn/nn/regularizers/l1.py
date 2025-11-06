import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.regularizers import BaseRegularizer
from LunarLearn.core import ops

xp = backend.xp

class L1(BaseRegularizer):
    """
    L1 Regularization.

    Adds a penalty proportional to the absolute value of each parameter tensor.

    Args:
        lam (float): Regularization strength (default: 1e-4)
    """
    def __init__(self, lam=1e-4, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = xp.array(lam, dtype=backend.DTYPE)

    def loss(self, param):
        """Compute L1 regularization loss for a single parameter tensor."""
        return ops.sum(ops.abs(param)) * self.lam