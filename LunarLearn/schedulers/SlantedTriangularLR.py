import LunarLearn.backend as backend
from LunarLearn.schedulers.BaseScheduler import BaseScheduler

xp = backend.xp
DTYPE = backend.DTYPE


class SlantedTriangularLR(BaseScheduler):
    def __init__(self, target, attr_name: str, max_epochs, cut_frac=0.1, ratio=32):
        """
        Slanted Triangular Learning Rate (STLR).

        Args:
            optimizer: optimizer object (with learning_rate0).
            max_epochs: total number of epochs.
            cut_frac: fraction of epochs used for increasing LR.
            ratio: ratio between max LR and min LR (max_lr / min_lr).
        """
        super().__init__(target, attr_name, mode="epoch")
        self.max_epochs = max_epochs
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.min_value = self.initial_value / ratio

        self.cut_epoch = int(self.max_epochs * self.cut_frac)

    def step(self, step=None):
        epoch = super().step(step)
        base_value = self.initial_value

        if epoch < self.cut_epoch:
            # Linearly increase from min_lr → base_lr
            factor = epoch / max(1, self.cut_epoch)
            new_value = self.min_value + factor * (base_value - self.min_value)
        else:
            # Linearly decrease from base_lr → min_lr
            factor = (epoch - self.cut_epoch) / max(1, self.max_epochs - self.cut_epoch)
            new_value = base_value - factor * (base_value - self.min_value)

        super().set_new_value(new_value)