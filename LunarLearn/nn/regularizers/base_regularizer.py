import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor

xp = backend.xp


class BaseRegularizer:
    """
    Base class for all regularizers.

    Defines the interface for computing regularization losses in an
    autograd-compatible way.

    Methods:
        __call__(model):
            Compute total regularization loss across all model parameters.
        loss(param):
            Compute regularization loss for a single parameter tensor.
    """
    def __init__(self, combine_mode="override"):
        if combine_mode not in ["override", "additive"]:
            raise ValueError("combine_mode must be 'override' or 'additive'")
        self.combine_mode = combine_mode

    def __call__(self, model):
        """
        Compute total regularization loss across all model parameters.

        Args:
            model: Any object exposing `parameters()` or
                   `parameters(with_layer=True)` returning dicts:
                   {"param": Tensor, "layer": Layer}

        Returns:
            Tensor: total regularization loss (autograd-compatible)
        """
        total_loss = None

        for param_desc in model.parameters(with_layer=True):
            p = param_desc["param"]
            layer = param_desc.get("layer", None)

            if not getattr(p, "requires_grad", False):
                continue

            # Base: global regularizer
            reg = self.loss(p)

            # Per-layer regularizer
            layer_reg = None
            if getattr(layer, "regularizer", None) is not None:
                layer_reg = layer.regularizer.loss(p)

            # Per-parameter regularizer
            param_reg = None
            if getattr(p, "regularizer", None) is not None:
                param_reg = p.regularizer.loss(p)

            # Combine logic
            if self.combine_mode == "additive":
                reg = sum(r for r in [reg, layer_reg, param_reg] if r is not None)
            else:  # override mode (layer > param > global)
                if param_reg is not None:
                    reg = param_reg
                elif layer_reg is not None:
                    reg = layer_reg
                else:
                    reg = reg

            # Accumulate
            if total_loss is None:
                total_loss = reg
            else:
                total_loss += reg

        if total_loss is None:
            return Tensor(0.0, requires_grad=False)

        return total_loss
    
    def loss(self, param):
        """Override in subclass."""
        return Tensor(0.0)

    def __add__(self, other):
        from LunarLearn.nn.regularizers import CompositeRegularizer
        return CompositeRegularizer([self, other])

    def __or__(self, other):
        return self.__add__(other)