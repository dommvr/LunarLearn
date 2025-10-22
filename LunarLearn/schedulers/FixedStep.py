import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class FixedStep(BaseScheduler):

    def __init__(self, target, attr_name: str, milestone, decay_rate):
        super().__init__(target, attr_name, mode='epoch')
        self.milestone = milestone
        self.decay_rate = decay_rate

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value
        new_value = base_value * (1/(1+self.decay_rate*xp.floor(epoch/self.milestone)))

        super().set_new_value(new_value)