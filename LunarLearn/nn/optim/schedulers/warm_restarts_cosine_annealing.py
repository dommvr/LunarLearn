import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.schedulers import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class WarmRestartsCosineAnnealing(BaseScheduler):

    def __init__(self, target, attr_name: str, min_value, max_value, milestone, multiplier=1):
        super().__init(target, attr_name, mode='epoch')
        self.min_value = min_value
        self.max_value = max_value
        self.milestone = milestone
        self.multiplier = multiplier

    def step(self, step=None):
        epoch = super().step(step)

        T_i = self.milestone
        t_curr = epoch
        while t_curr >= T_i:
            t_curr -= T_i
            T_i *= self.multiplier
        factor = 0.5 * (1 + xp.cos(xp.pi * t_curr / T_i))
        new_value = self.min_value + (self.min_value - self.min_value) * factor
        
        super().set_new_value(new_value)