import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.regularizers import BaseRegularizer
from LunarLearn.core import Tensor, ops

xp = backend.xp


class Contractive(BaseRegularizer):
    """
    Contractive Regularizer (used in Contractive Autoencoders).

    Penalizes sensitivity of hidden representation wrt input.
    Loss = lam * ||dA/dX||^2
    Assumes the layer provides `layer.dA_dX` (Jacobian norm estimate).
    """
    def __init__(self, lam=1e-4, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = lam

    def loss(self, param):
        # no direct param regularization here
        return Tensor(0.0)

    def __call__(self, layer):
        if hasattr(layer, "dA_dX") and layer.dA_dX is not None:
            return ops.sum(layer.dA_dX ** 2) * self.lam
        return Tensor(0.0)
