import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class PolynomialDecay(BaseScheduler):
    def __init__(self, target, attr_name: str, max_epochs, power=1.0, end_value=0.0):
        """
        Polynomial decay scheduler.

        Args:
            optimizer: optimizer with .learning_rate0
            max_epochs: total number of epochs for decay
            power: polynomial power (1 = linear decay, 2 = quadratic decay)
            end_lr: minimum learning rate after decay
        """
        super().__init__(target, attr_name, mode='epoch')
        self.max_epochs = max_epochs
        self.power = power
        self.end_value = end_value

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value
        factor = (1 - min(epoch, self.max_epochs) / self.max_epochs) ** self.power
        new_value = self.end_value + (base_value - self.end_value) * factor

        super().set_new_value(new_value)
