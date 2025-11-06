import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.schedulers import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class WarmCosineAnnealing(BaseScheduler):
    def __init__(self, target, attr_name: str, warmup_epochs, max_epochs, min_value=0.0):
        super().__init__(target, attr_name, mode='epoch')
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_value = min_value

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        if epoch < self.warmup_epochs:
            # Linear warmup: from 0 → base_lr
            factor = epoch / float(self.warmup_epochs)
            new_value = base_value * factor
        else:
            # Cosine decay: from base_lr → min_lr
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            cos_inner = xp.pi * progress
            factor = 0.5 * (1 + xp.cos(cos_inner))
            new_value = self.min_value + (base_value - self.min_value) * factor

        super().set_new_value(new_value)

