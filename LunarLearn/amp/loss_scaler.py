import LunarLearn.backend as backend
from LunarLearn.tensor import Tensor

xp = backend.xp

class BaseLossScaler:
    """
    Base class for loss scaling in mixed-precision training.

    Loss scaling helps prevent underflow in float16 computations by multiplying
    the loss by a large scale factor before backpropagation, then dividing gradients
    by the same factor before the optimizer step.

    This class provides core gradient handling utilities used by static and dynamic
    scalers.
    """
    def __init__(self):
        self.scale = None

    def get_config(self):
        from LunarLearn.engine import serialize_value

        init_params = ["init_scale", "scale_factor", "min_scale", "max_scale", "step"]

        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "params": {
                k: serialize_value(v)
                for k, v in self.__dict__.items()
                if k in init_params
            },
            "extra": {
                k: serialize_value(v)
                for k, v in self.__dict__.items()
                if k not in init_params and k != "layers"
            }
        }

    @classmethod
    def from_config(cls, config):
        import importlib

        # Import the module and class
        module = importlib.import_module(config["module"])
        klass = getattr(module, config["class"])

        # Initialize object with params
        obj = klass(**config.get("params", {}))

        # Set extra attributes after init
        for k, v in config.get("extra", {}).items():
            setattr(obj, k, v)

        return obj

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Return scaled loss Tensor."""
        return loss * self.scale

    def _iter_grads(self, model):
        """Yield all parameter gradients from the model."""
        for param_desc in model.parameters(with_layer=True):
            p = param_desc["param"] if isinstance(param_desc, dict) else param_desc
            if getattr(p, "grad", None) is not None:
                yield p.grad

    def check_if_safe(self, model) -> bool:
        """
        Check if gradients contain NaN or Inf values.
        Returns False if any gradient is non-finite.
        """
        for g in self._iter_grads(model):
            if g is None:
                continue
            # scalar reduction (fast, minimal host sync)
            s = xp.sum(g.data)
            if not xp.isfinite(s):
                return False
        return True
    
    def unscale_grads(self, model) -> bool:
        """
        Unscale gradients in place.
        If grads are non-finite, decreases the scale and skips optimizer step.
        Returns:
            bool: True if gradients are finite (safe to proceed with optimizer step),
                  False if overflow was detected (optimizer step should be skipped).
        """
        if self.check_if_safe(model):
            inv_scale = 1.0 / self.scale

            # Unscale gradients in-place
            for g in self._iter_grads(model):
                if g is not None:
                    g *= inv_scale


class StaticLossScaler(BaseLossScaler):
    """
    Static loss scaler for mixed-precision training.

    Uses a fixed scaling factor throughout training. This is simpler and slightly
    faster than dynamic scaling, but may not adapt well if gradient magnitudes vary
    significantly during training.
    """
    def __init__(self, scale=1024):
        super().__init__()
        self.scale = scale


class DynamicLossScaler(BaseLossScaler):
    """
    Dynamic loss scaler for mixed-precision training.

    Automatically adjusts the scale factor based on gradient statistics.
    If overflow (NaN/Inf) is detected, the scale is reduced. If several consecutive
    steps are successful, the scale is increased — allowing better utilization of
    float16 range over time.

    This approach is more robust than static scaling and recommended in most cases.
    """
    def __init__(self, init_scale=1024, scale_factor=2, min_scale=1, max_scale=2**16, step=5):
        self.scale = float(init_scale)
        self.temp_scale = self.scale
        self.scale_factor = float(scale_factor)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.step = step
        self.good_steps = 0

    def unscale_grads(self, model) -> bool:
        """
        Unscale gradients and dynamically adjust the scaling factor.

        - If any gradient is NaN or Inf, the scale is decreased and the optimizer step is skipped.
        - If gradients are finite for a number of steps (`step`), the scale is increased.

        Returns:
            bool: True if gradients are finite (safe to proceed with optimizer step),
                  False if overflow was detected (optimizer step should be skipped).
        """
        if not self.check_if_safe(model):
            self.good_steps = 0
            old_scale = self.scale
            self.scale = max(self.min_scale, self.scale / self.scale_factor)
            print(f"[AMP] Overflow detected, reducing scale from {old_scale:.2e} to {self.scale:.2e}")
            return False

        self.good_steps += 1
        inv_scale = 1.0 / self.scale

        # Unscale gradients in-place
        for g in self._iter_grads(model):
            if g is not None:
                g *= inv_scale

        # Increase scale after a number of good steps
        if (self.good_steps % self.step) == 0:
            self.scale = min(self.scale * self.scale_factor, self.max_scale)

        return True