import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE


class OneCycleLR(BaseScheduler):
    def __init__(
        self,
        target,
        attr_name: str,
        max_value,
        total_steps,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1e4,
        max_momentum=0.95,
        min_momentum=0.85,
    ):
        """
        OneCycleLR Scheduler.

        Args:
            optimizer: Optimizer object (with learning_rate and learning_rate0).
            max_lr: The peak learning rate reached during training.
            total_steps: Total number of steps (epochs * steps_per_epoch, or manually passed).
            pct_start: Fraction of total_steps where LR increases from initial_lr to max_lr.
            div_factor: Determines initial_lr = max_lr / div_factor.
            final_div_factor: Determines min_lr = initial_lr / final_div_factor.
            max_momentum: Starting (high) momentum value.
            min_momentum: Lowest momentum value during cycle.
        """
        super().__init__(target, attr_name, mode="step")
        self.max_value = float(max_value)
        self.total_steps = int(total_steps)
        self.pct_start = float(pct_start)
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)
        self.max_momentum = float(max_momentum)
        self.min_momentum = float(min_momentum)

        # Derived values
        self.initial_value = self.max_value / self.div_factor
        self.min_value = self.initial_value / self.final_div_factor

        super().set_new_value(self.initial_value)
        if hasattr(self.target, "momentum"):
            self.target.momentum = self.max_momentum

    def step(self, step=None):
        step = super().step(step)
        progress = step / float(self.total_steps)
        base_value = self.initial_value

        # -------------------
        # Learning Rate
        # -------------------
        if progress <= self.pct_start:
            # Phase 1: increase LR linearly
            scale = progress / self.pct_start
            new_value = base_value + (self.max_value - base_value) * scale
        else:
            # Phase 2: cosine decay
            scale = (progress - self.pct_start) / (1 - self.pct_start)
            new_value = self.min_value + 0.5 * (self.max_value - self.min_value) * (1 + xp.cos(xp.pi * scale))

        super().set_new_value(new_value)

        # -------------------
        # Momentum
        # -------------------
        if hasattr(self.target, "momentum"):
            if progress <= self.pct_start:
                # Phase 1: momentum decreases linearly (inverse to LR)
                scale = progress / self.pct_start
                mom = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
            else:
                # Phase 2: momentum increases back via cosine
                scale = (progress - self.pct_start) / (1 - self.pct_start)
                mom = self.min_momentum + 0.5 * (self.max_momentum - self.min_momentum) * (1 + xp.cos(xp.pi * scale))

            self.target.momentum = float(mom)