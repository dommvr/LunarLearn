import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.schedulers import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class LinearWarmup(BaseScheduler):
    def __init__(self, target, attr_name: str, warmup_epochs, total_epochs, min_value=0.0):
        """
        Linear warmup followed by linear decay.

        optimizer: optimizer object
        warmup_epochs: number of epochs to linearly increase LR
        total_epochs: total number of epochs for decay
        min_lr: minimum learning rate at the end of decay
        """
        super().__init__(target, attr_name, mode='epoch')
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_value = min_value

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        if epoch < self.warmup_epochs:
            # Linear warmup: LR increases from 0 (or min_lr) to initial LR
            factor = epoch / self.warmup_epochs
            new_value = self.min_value + factor * (base_value - self.min_value)
        else:
            # Linear decay after warmup
            decay_epochs = self.total_epochs - self.warmup_epochs
            factor = max(0.0, (self.total_epochs - epoch) / decay_epochs)
            new_value = self.min_value + factor * (base_value - self.min_value)

        super().set_new_value(new_value)
