import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class GradualWarmup(BaseScheduler):
    def __init__(self, target, attr_name: str, warmup_epochs, multiplier=1.0, after_scheduler=None):
        """
        Gradual Warmup Scheduler

        Args:
            optimizer: optimizer object (with learning_rate and learning_rate0).
            warmup_epochs: number of epochs to warm up.
            multiplier: target factor for LR (1.0 -> warmup to base_lr,
                        >1.0 -> warmup to base_lr * multiplier).
            after_scheduler: optional scheduler to apply after warmup.
        """
        super().__init__(target, attr_name, mode='epoch')
        self.warmup_epochs = warmup_epochs
        self.multiplier = multiplier
        self.after_scheduler = after_scheduler
        self.finished = False

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        if epoch < self.warmup_epochs:
            # Linearly scale LR from base_lr to base_lr * multiplier
            factor = (1.0 + (self.multiplier - 1.0) * (epoch / self.warmup_epochs))
            new_value = base_value * factor
        else:
            if self.after_scheduler is not None:
                # Hand over control to after_scheduler
                self.after_scheduler.step(epoch - self.warmup_epochs)
                new_value = getattr(self.after_scheduler.target, self.attr_name)
                self.finished = True
            else:
                new_value = base_value * self.multiplier

        super().set_new_value(new_value)
