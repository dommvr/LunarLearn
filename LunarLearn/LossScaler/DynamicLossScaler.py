import LunarLearn.backend as backend
from LunarLearn.LossScaler.BaseLossScaler import BaseLossScaler

xp = backend.xp
DTYPE = backend.DTYPE

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

        self.model = None

    def unscale_grads(self) -> bool:
        """
        Unscale gradients and dynamically adjust the scaling factor.

        - If any gradient is NaN or Inf, the scale is decreased and the optimizer step is skipped.
        - If gradients are finite for a number of steps (`step`), the scale is increased.

        Returns:
            bool: True if gradients are finite (safe to proceed with optimizer step),
                  False if overflow was detected (optimizer step should be skipped).
        """
        if not self.check_if_safe():
            self.good_steps = 0
            self.scale = max(self.min_scale, self.scale / self.scale_factor)
            return False

        self.good_steps += 1
        inv_scale = 1.0 / self.scale

        # Unscale gradients in-place
        for g in self._iter_grads():
            if g is not None:
                g *= inv_scale

        # Increase scale after a number of good steps
        if (self.good_steps % self.step) == 0:
            self.scale = min(self.scale * self.scale_factor, self.max_scale)

        return True