from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer
from LunarLearn.tensor import Tensor

class ActivationHookRegularizer(BaseRegularizer):
    """
    Flexible user-defined regularizer on layer activations, weights, or attributes.

    Example:
        reg = ActivationHookRegularizer(lambda A: 1e-4 * ops.mean(A ** 2))
        dense_layer.regularizer = reg
    """
    def __init__(self, func):
        """
        Args:
            func (callable): penalty function, either:
                func(layer) -> Tensor
                or func(A) -> Tensor
        """
        super().__init__()
        self.func = func

    def loss(self, param):
        return Tensor(0.0)

    def __call__(self, layer):
        # Try full-layer callable first
        try:
            return self.func(layer)
        except TypeError:
            # Fallback: assume callable expects activations only
            if hasattr(layer, "A") and layer.A is not None:
                return self.func(layer.A)
            return Tensor(0.0)