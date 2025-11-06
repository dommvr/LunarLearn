import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.schedulers import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE


class CosinePowerAnnealing(BaseScheduler):
    def __init__(self, target, attr_name: str, max_epochs, min_value=0.0, power=1.0):
        """
        Cosine Power Annealing.

        Args:
            optimizer: Optimizer object (with learning_rate0).
            max_epochs: Total number of epochs for training.
            min_lr: Final learning rate after decay.
            power: Exponent applied to the cosine term.
                   - power=1.0 → standard cosine annealing
                   - power>1.0 → sharper decay
                   - power<1.0 → flatter decay
        """
        super().__init__(target, attr_name, mode="epoch")
        self.max_epochs = max_epochs
        self.min_value = min_value
        self.power = power

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        # Progress from 0 to 1
        progress = min(epoch, self.max_epochs) / float(self.max_epochs)

        # Cosine annealing factor
        cos_inner = xp.pi * progress
        factor = 0.5 * (1 + xp.cos(cos_inner))

        # Apply power transformation
        factor = factor ** self.power

        # New value
        new_value = self.min_value + (base_value - self.min_value) * factor

        super().set_new_value(new_value)