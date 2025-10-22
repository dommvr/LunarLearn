import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class ExponentialWarmup(BaseScheduler):
    def __init__(self, target, attr_name, warmup_epochs, max_epochs=None, min_value=0.0):
        """
        Exponential Warmup (optionally followed by flat decay).
        
        Args:
            optimizer: optimizer object (with learning_rate and learning_rate0).
            warmup_epochs: number of warmup epochs.
            max_epochs: optional, if given -> learning rate can decay after warmup.
            min_lr: minimum learning rate after decay (if max_epochs provided).
        """
        super().__init__(target, attr_name, mode='epoch')
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_value = min_value

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        if epoch < self.warmup_epochs:
            # Exponential warmup: start small, ramp up fast
            # lr = base_lr * (exp(epoch / warmup) - 1) / (e - 1)
            factor = (xp.exp(epoch / self.warmup_epochs) - 1.0) / (xp.e - 1.0)
            new_value = base_value * factor
        else:
            # Optionally keep flat or decay to min_lr
            if self.max_epochs is None:
                new_value = base_value
            else:
                progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
                new_value = self.min_value + (base_value - self.min_value) * (1 - progress)

        super().set_new_value(new_value)
