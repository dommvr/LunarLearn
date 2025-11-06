import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.schedulers import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE


class TanhDecay(BaseScheduler):
    def __init__(self, target, attr_name: str, max_epochs, min_value=0.0, alpha=5.0, reverse=False):
        """
        Hyperbolic Tangent (tanh) decay or warmup.

        Args:
            optimizer: Optimizer object (with learning_rate0).
            max_epochs: Total number of epochs for training.
            min_lr: Minimum learning rate at the end of decay (or start of warmup).
            alpha: Steepness of the tanh curve (higher = faster transition).
            reverse: If True, schedule becomes a tanh warmup 
                     (min_lr → base_lr instead of base_lr → min_lr).
        """
        super().__init__(target, attr_name, mode="epoch")
        self.max_epochs = max_epochs
        self.min_value = min_value
        self.alpha = alpha
        self.reverse = reverse

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        # Normalize progress in [0,1]
        progress = epoch / float(self.max_epochs)

        # Base tanh factor: smoothly goes from ~1 → 0
        factor = 0.5 * (1 - xp.tanh(self.alpha * (progress - 1)))

        if self.reverse:
            # Warmup: min_lr → base_lr
            new_value = self.min_value + (base_value - self.min_value) * (1 - factor)
        else:
            # Decay: base_lr → min_lr
            new_value = self.min_value + (base_value - self.min_value) * factor

        super().set_new_value(new_value)