import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.schedulers import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE


class Noam(BaseScheduler):
    def __init__(self, target, attr_name: str, model_dim, warmup_steps=4000, factor=1.0):
        """
        Noam Scheduler (used in Transformers).

        Args:
            optimizer: Optimizer object (with learning_rate and learning_rate0).
            model_dim: Dimensionality of the model (d_model).
            warmup_steps: Number of warmup steps before decay begins.
            factor: Scaling factor for the learning rate (default=1.0).
        """
        super().__init__(target, attr_name, mode="step")
        self.model_dim = float(model_dim)
        self.warmup_steps = int(warmup_steps)
        self.factor = float(factor)

    def step(self, step=None):
        step = super().step(step)

        scale = self.model_dim ** -0.5
        new_value = self.factor * scale * min(step ** -0.5,
                                       step * (self.warmup_steps ** -1.5))

        super().set_new_value(new_value)

