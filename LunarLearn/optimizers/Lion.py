import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Lion(BaseOptimizer):
    """
    Lion optimizer with autograd support.

    This optimizer uses momentum tracking combined with the sign of the 
    momentum for parameter updates. It is designed to reduce memory usage 
    and improve convergence stability compared to Adam-like methods.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.01.
        beta1 (float, optional): 
            Exponential decay rate for momentum accumulation. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for momentum decay. Default is 0.99.

    Attributes:
        state (dict): 
            Per-parameter state storing first-moment estimates.
    """
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.state = {}

    def step(self, params):
        for param_desc in params:
            # --- Extract param & layer if available ---
            if isinstance(param_desc, dict):
                p = param_desc["param"]
                layer = param_desc.get("layer", None)
            else:
                p = param_desc
                layer = None

            if not isinstance(p, Tensor) or not p.requires_grad:
                continue
            if p.grad is None:
                continue

            grad = p.grad
            lr = self._get_lr(param_desc)

            # Initialize state if needed
            if p not in self.state:
                self.state[p] = {
                    "m": xp.zeros_like(p.data, dtype=DTYPE),  # momentum
                }

            state = self.state[p]

            # Update momentum (EMA of gradients)
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            # Parameter update: sign of momentum
            p.data -= lr * xp.sign(state["m"])

            # Decay momentum
            state["m"] *= self.beta2