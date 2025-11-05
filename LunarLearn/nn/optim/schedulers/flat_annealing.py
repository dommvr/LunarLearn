import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class FlatAnnealing(BaseScheduler):
    def __init__(self, target, attr_name: str, max_epochs, flat_fraction=0.3, min_value=0.0):
        """
        Ranger's Flat + Anneal LR Scheduler.

        Args:
            optimizer: optimizer object.
            max_epochs: total number of epochs.
            flat_fraction: fraction of max_epochs to keep LR flat.
            min_lr: minimum LR after annealing.
        """
        super().__init__(target, attr_name, mode='epoch')
        self.max_epochs = max_epochs
        self.flat_fraction = flat_fraction
        self.min_value = min_value
        self.flat_epochs = int(flat_fraction * max_epochs)

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        if epoch < self.flat_epochs:
            # Flat phase
            new_value = base_value
        else:
            # Cosine annealing phase
            progress = (epoch - self.flat_epochs) / max(1, self.max_epochs - self.flat_epochs)
            cos_inner = xp.pi * progress
            factor = 0.5 * (1 + xp.cos(cos_inner))
            new_value = self.min_value + (base_value - self.min_value) * factor

        super().set_new_value(new_value)