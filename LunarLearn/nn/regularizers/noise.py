import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.regularizers import BaseRegularizer
from LunarLearn.core import Tensor, ops

xp = backend.xp


class Noise(BaseRegularizer):
    """
    Noise-based regularizer for robustness and smoothness.

    Modes:
        - "additive": adds Gaussian noise and applies penalty to magnitude.
        - "consistency": penalizes difference between clean and noisy activations.

    Typical usage:
        layer.regularizer = NoiseRegularizer(lam=1e-3, sigma=0.1, mode="consistency")
    """
    def __init__(self, lam=1e-3, sigma=0.1, mode="consistency", combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.lam = lam
        self.sigma = sigma
        self.mode = mode.lower()

    def loss(self, param):
        """No weight-level penalty in this regularizer."""
        return Tensor(0.0)

    def __call__(self, layer):
        """
        Apply noise regularization based on layer activations.

        Args:
            layer: A layer object with `.A` (activation output).

        Returns:
            Tensor: autograd-compatible regularization loss.
        """
        if not hasattr(layer, "A") or layer.A is None:
            return Tensor(0.0)

        A = layer.A
        noise = xp.random.normal(0.0, self.sigma, size=A.shape).astype(A.dtype)

        if self.mode == "additive":
            # Simple additive noise penalty: lam * mean(|A + noise|)
            noisy_A = A + noise
            return self.lam * ops.mean(ops.abs(noisy_A))

        elif self.mode == "consistency":
            # Penalize difference between clean and noisy outputs
            noisy_A = A + noise
            diff = noisy_A - A
            return self.lam * ops.mean(diff ** 2)

        else:
            raise ValueError(f"Unknown noise regularizer mode: {self.mode}")
