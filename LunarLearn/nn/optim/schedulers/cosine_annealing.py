import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class CosineAnnealing(BaseScheduler):
    def __init__(self, target, attr_name: str, max_epochs, min_value=0.0):
        """
        Cosine annealing learning rate scheduler.

        Args:
            optimizer: optimizer with .learning_rate0
            max_epochs: number of epochs to reach min_lr
            min_lr: minimum learning rate at the end of the cycle
        """
        super().__init__(target, attr_name, mode='epoch')
        self.max_epochs = max_epochs
        self.min_value = min_value

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        cos_inner = xp.pi * min(epoch, self.max_epochs) / self.max_epochs
        factor = 0.5 * (1 + xp.cos(cos_inner))
        new_value = self.min_value + (base_value - self.min_value) * factor

        super().set_new_value(new_value)