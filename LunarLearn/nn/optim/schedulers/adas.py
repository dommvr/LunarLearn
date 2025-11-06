import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.schedulers import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE

class AdaS(BaseScheduler):
    """
    Adaptive Scheduling (AdaS) LR scheduler with dynamic increase/decrease.
    - Increases LR when loss is decreasing fast.
    - Decreases LR when loss stagnates or increases.
    """
    def __init__(self, target, attr_name: str, min_value=1e-5, max_value=1e-2, patience=3,
                 factor_down=0.5, factor_up=1.1, mode='epoch'):
        super().__init__(target, attr_name, mode=mode)
        self.min_value = min_value
        self.max_value = max_value
        self.patience = patience
        self.factor_down = factor_down  # reduction factor
        self.factor_up = factor_up      # increase factor
        self.best_loss = float('inf')
        self.num_bad_epochs = 0

    def step(self, current_loss=None, step=None):
        _ = super().step(step)
        base_value = self.initial_value

        if current_loss is None:
            return base_value

        if current_loss < self.best_loss:
            # Loss improved → possibly increase LR
            self.best_loss = current_loss
            self.num_bad_epochs = 0
            new_value = min(base_value * self.factor_up, self.max_value)
        else:
            # Loss did not improve
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                # Reduce LR if no improvement for `patience` steps
                new_value = max(base_value * self.factor_down, self.min_value)
                self.num_bad_epochs = 0  # reset counter

        super().set_new_value(new_value)