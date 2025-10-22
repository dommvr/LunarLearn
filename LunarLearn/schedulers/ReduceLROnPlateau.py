import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class ReduceLROnPlateau(BaseScheduler):
    def __init__(self, target, attr_name: str, mode='min', factor=0.1, patience=10, min_value=1e-6, threshold=1e-4, cooldown=0):
        """
        Reduce learning rate when a metric has stopped improving.

        Args:
            optimizer: optimizer object.
            mode: 'min' or 'max', determines if metric should be minimized or maximized.
            factor: factor by which the learning rate will be reduced.
            patience: number of epochs with no improvement to wait before reducing LR.
            min_lr: lower bound on LR.
            threshold: minimum change to consider as improvement.
            cooldown: number of epochs to wait after a LR change before monitoring again.
        """
        super().__init__(target, attr_name, mode='epoch')
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_value = min_value
        self.threshold = threshold
        self.cooldown = cooldown

        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

    def step(self, current_metric):
        base_value = self.initial_value
        # First epoch setup
        if self.best is None:
            self.best = current_metric
            return getattr(self.target, self.attr_name)

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # reset counter during cooldown

        improved = ((self.mode == 'min' and current_metric < self.best - self.threshold) or
                    (self.mode == 'max' and current_metric > self.best + self.threshold))

        if improved:
            self.best = current_metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            old_value = super().get_value()
            new_value = max(old_value * self.factor, self.min_value)
            if new_value < old_value:
                super().set_new_value(new_value)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0  # reset counter after LR reduction
