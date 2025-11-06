from LunarLearn.nn.regularizers import BaseRegularizer
from LunarLearn.core import Tensor


class CompositeRegularizer(BaseRegularizer):
    """
    Combines multiple regularizers into a single one.
    Supports autograd and layer-level detection.

    Example:
        reg = L1(1e-4) + L2(1e-4)
        loss += reg(model)
    """
    def __init__(self, regularizers, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.regularizers = []

        # Flatten nested composites
        for r in regularizers:
            if isinstance(r, CompositeRegularizer):
                self.regularizers.extend(r.regularizers)
            else:
                self.regularizers.append(r)

    def loss(self, param):
        """Sum all component regularizer losses for a parameter."""
        total_loss = None

        for r in self.regularizers:
            reg_loss = r.loss(param)
            if total_loss is None:
                total_loss = reg_loss
            else:
                total_loss += reg_loss

        if total_loss is None:
            return Tensor(0.0)
        return total_loss

    def __call__(self, model):
        """Apply composite regularizer across all model parameters."""
        total_loss = None

        for r in self.regularizers:
            reg_loss = r(model)
            if total_loss is None:
                total_loss = reg_loss
            else:
                total_loss += reg_loss

        if total_loss is None:
            return Tensor(0.0)
        return total_loss