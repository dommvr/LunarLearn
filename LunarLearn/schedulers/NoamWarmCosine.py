import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE


class NoamWarmCosine(BaseScheduler):
    def __init__(self, target, attr_name: str, model_dim, warmup_steps=4000, max_steps=100000,
                 min_value=0.0, factor=1.0):
        """
        Noam + Cosine Decay Scheduler.
        - Uses Noam warmup strategy, then switches to cosine annealing.

        Args:
            optimizer: Optimizer object (with learning_rate and learning_rate0).
            model_dim: Model dimensionality (d_model).
            warmup_steps: Number of warmup steps.
            max_steps: Total training steps (for cosine schedule).
            min_lr: Minimum learning rate after decay.
            factor: Scaling factor.
        """
        super().__init__(target, attr_name, mode="step")
        self.model_dim = float(model_dim)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.min_value = float(min_value)
        self.factor = float(factor)

    def step(self, step=None):
        step = super().step(step)

        # Noam warmup formula
        scale = self.model_dim ** -0.5
        noam_value = self.factor * scale * step * (self.warmup_steps ** -1.5)

        if step < self.warmup_steps:
            new_value = noam_value
        else:
            # Progress after warmup
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            cos_inner = xp.pi * min(1.0, progress)
            factor = 0.5 * (1 + xp.cos(cos_inner))
            new_value = self.min_value + (noam_value - self.min_value) * factor

        super().set_new_value(new_value)