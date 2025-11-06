import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.regularizers import BaseRegularizer
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE


class SharpnessAware(BaseRegularizer):
    """
    First-order Hessian trace proxy regularizer (SAM-style).

    Encourages parameters to converge to flatter minima by penalizing
    high gradient norms — an indirect estimate of local curvature.

    Unlike HessianTraceRegularizer, this requires only first-order grads.

    Concept:
        penalty ≈ lam * ||∇_θ L||²

    Args:
        lam (float): Strength of the penalty term.
        with_layer (bool): Whether to use model.parameters(with_layer=True)
                           for consistency with optimizer logic.
        detach (bool): If True, detaches grad before computing its norm
                       (prevents 2nd-order computation through grad).
    """

    def __init__(self, lam: float = 1e-4, with_layer: bool = True, detach: bool = True):
        self.lam = lam
        self.with_layer = with_layer
        self.detach = detach

    def __call__(self, model):
        """
        Compute lam * sum(||grad||²) across all trainable parameters.

        Must be called *after backward() or inside training step* where
        gradients are available.
        """
        total = None

        for param_desc in model.parameters(with_layer=self.with_layer):
            p = param_desc["param"] if isinstance(param_desc, dict) else param_desc
            if p.grad is None:
                continue

            g = p.grad
            if self.detach:
                g = ops.stop_gradient(g)  # equivalent to .detach() — no 2nd-order graph

            gn = ops.sum(g * g)
            total = gn if total is None else (total + gn)

        if total is None:
            return Tensor(0.0)

        return total * self.lam
