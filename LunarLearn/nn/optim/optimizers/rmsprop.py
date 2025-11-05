import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class RMSProp(BaseOptimizer):
    """
    RMSProp optimizer with autograd support.

    RMSProp maintains a moving average of squared gradients to adaptively
    scale the learning rate per parameter. It helps stabilize training,
    especially in recurrent and deep networks.

    Args:
        learning_rate (float, optional):
            Base learning rate. Default is 0.001.
        beta (float, optional):
            Exponential decay rate for squared gradient average. Default is 0.9.
        epsilon (float, optional):
            Small constant for numerical stability. Default is 1e-8.
        weight_decay (float, optional):
            Decoupled weight decay coefficient. Default is 0.

    Attributes:
        state (dict):
            Per-parameter state storing the exponential moving average
            of squared gradients.
    """
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta = xp.array(beta, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.weight_decay = xp.array(weight_decay, dtype=DTYPE)
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
                    "v": xp.zeros_like(p.data, dtype=DTYPE),  # EMA of squared gradients
                }

            state = self.state[p]

            # Update squared gradient average
            state["v"] = self.beta * state["v"] + (1 - self.beta) * (grad * grad)

            # Update rule
            denom = xp.sqrt(state["v"]) + self.epsilon
            update = grad / denom

            if self.weight_decay > 0:
                update += self.weight_decay * p.data

            p.data -= lr * update