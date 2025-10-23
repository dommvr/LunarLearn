from LunarLearn.amp.BaseLossScaler import BaseLossScaler

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
    