import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class ExponentialDecay(BaseScheduler):
    def __init__(self, target, attr_name: str, decay_rate=None, decay_constant=None):
        """
        Exponential decay scheduler.

        Args:
            optimizer: optimizer with .learning_rate0
            decay_rate: multiplicative decay per epoch (lr = lr0 * decay_rate^epoch)
            decay_constant: k for continuous decay (lr = lr0 * exp(-k*epoch))
        """
        super().__init__(target, attr_name, mode='epoch')
        self.decay_rate = decay_rate
        self.decay_constant = decay_constant

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        if self.decay_rate is not None:
            new_value = base_value * (self.decay_rate ** epoch)
        elif self.decay_constant is not None:
            new_value = base_value * xp.exp(-self.decay_constant * epoch)
        else:
            raise ValueError("Must specify decay_rate or decay_constant")
        
        super().set_new_value(new_value)