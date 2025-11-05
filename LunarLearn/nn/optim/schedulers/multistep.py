import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class MultiStep(BaseScheduler):

    def __init__(self, target, attr_name: str, milestones, decay_rate):
        """
        MultiStep LR Scheduler

        Args:
            optimizer: optimizer object with .learning_rate0
            milestones: list of epochs where LR decays
            decay_rate: multiplicative factor at each milestone
        """
        super().__init__(target, attr_name, mode='epoch')
        self.milestones = sorted(milestones)
        self.decay_rate = decay_rate

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value
        n_decays = sum(epoch >= m for m in self.milestones)
        new_value = base_value * (self.decay_rate ** n_decays)

        super().set_new_value(new_value)