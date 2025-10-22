import LunarLearn.backend as backend
from LunarLearn.tensor import ops
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer

xp = backend.xp

class InformationBottleneck(BaseRegularizer):
    """
    Variational Information Bottleneck (VIB) regularizer.
    Encourages compressed latent representations.

    Assumes the model (or a layer) exposes parameters:
        mu: mean of latent distribution
        logvar: log-variance of latent distribution

    Loss = λ * KL(q(z|x) || N(0, I))
         = -0.5 * λ * Σ(1 + logvar - μ² - exp(logvar))
    """
    def __init__(self, lam=1e-4, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = lam

    def loss(self, mu, logvar):
        kl = -0.5 * ops.sum(1 + logvar - mu**2 - ops.exp(logvar))
        return self.lam * kl