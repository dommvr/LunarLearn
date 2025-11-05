import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class CyclicLR(BaseScheduler):
    def __init__(self, target, attr_name: str, base_value, max_value, step_size, mode="triangular", gamma=1.0):
        """
        Cyclical Learning Rate Scheduler

        Args:
            optimizer: optimizer object with learning_rate and learning_rate0
            base_lr: lower bound of the cycle
            max_lr: upper bound of the cycle
            step_size: number of iterations per half-cycle
            mode: one of ["triangular", "triangular2", "exp_range"]
            gamma: decay factor for 'exp_range' mode
        """
        super().__init__(target, attr_name, mode='step')
        self.base_value = base_value
        self.max_value = max_value
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma

    def step(self, step=None):
        iteration = super().step(step)
        # Compute cycle index
        cycle = xp.floor(1 + iteration / (2 * self.step_size))
        x = xp.abs(iteration / self.step_size - 2 * cycle + 1)

        # Scale factor between 0 and 1
        scale = xp.maximum(0, 1 - x)

        # Compute value delta
        value_delta = (self.max_value - self.base_value) * scale

        # Mode scaling
        if self.mode == "triangular":
            factor = 1.0
        elif self.mode == "triangular2":
            factor = 1 / (2.0 ** (cycle - 1))
        elif self.mode == "exp_range":
            factor = self.gamma ** iteration
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # New value
        new_value = self.base_value + value_delta * factor
        super().set_new_value(new_value)