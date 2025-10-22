import LunarLearn.backend as backend
from LunarLearn.tensor import ops
from LunarLearn.tensor import Tensor
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer

xp = backend.xp

class MaxNorm(BaseRegularizer):
    """
    Max-Norm Constraint Regularizer.
    Projects parameters onto a max-norm ball during optimization.

    Note: This is a constraint, not a penalty — returns zero loss.
    """
    def __init__(self, max_norm=3.0, axis=0, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.max_norm = xp.array(max_norm, dtype=backend.DTYPE)
        self.axis = axis

    def loss(self, param):
        # No scalar penalty term — handled by projection after update.
        return Tensor(0.0)

    def project(self, param):
        """Applies max-norm constraint directly to the parameter."""
        norms = ops.sqrt(ops.sum(param ** 2, axis=self.axis, keepdims=True)) + 1e-8
        desired = ops.clip(norms, 0, self.max_norm)
        scale = desired / norms
        return param * scale