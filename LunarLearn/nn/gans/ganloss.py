from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import ops
from LunarLearn.nn.gans.utils import vanilla, lsgan, hinge, wasserstein

MODES = {
    "vanilla": vanilla,
    "lsgan": lsgan,
    "hinge": hinge,
    "wasserstein": wasserstein
}

class GANLoss(BaseLoss):
    def __init__(self, mode="vanilla"):
        super().__init__(trainable=False)
        if mode not in MODES:
            raise ValueError(f"Unsupported loss mode '{mode}'. "
                             f"Available: {list(MODES.keys())}")
        else:
            self.loss_fn = MODES[mode]

        self.mode = mode

    def forward(self, d_real, d_fake):
        d_loss, g_loss = self.loss_fn(d_real, d_fake)
        return d_loss, g_loss