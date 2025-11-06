from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer
from LunarLearn.tensor import ops
from LunarLearn.tensor import Tensor

class Activity(BaseRegularizer):
    """
    Penalizes large activations.
    Loss = lam * mean(|A|) or lam * mean(A^2)
    Similar to Keras' `activity_regularizer`.
    """
    def __init__(self, lam=1e-4, mode="l1", combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = lam
        self.mode = mode.lower()

    def loss(self, param):
        # not applied on weights directly
        return Tensor(0.0)

    def __call__(self, layer):
        if not hasattr(layer, "A") or layer.A is None:
            return Tensor(0.0)

        if self.mode == "l1":
            return self.lam * ops.mean(ops.abs(layer.A))
        elif self.mode == "l2":
            return self.lam * ops.mean(layer.A ** 2)
        else:
            raise ValueError(f"Unknown activity regularizer mode: {self.mode}")