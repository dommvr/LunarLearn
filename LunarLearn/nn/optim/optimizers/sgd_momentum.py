import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class SGDMomentum(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) with momentum.

    This optimizer accumulates a velocity vector in directions of persistent 
    reduction in the objective across iterations, helping accelerate 
    convergence and smooth updates.

    Args:
        learning_rate (float, optional): 
            Step size for parameter updates. Default is 0.01.
        beta (float, optional): 
            Momentum coefficient. Higher values give more weight to past 
            gradients. Default is 0.9.

    Attributes:
        state (dict): 
            Per-parameter state storing the momentum buffer.
    """
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = xp.array(beta, dtype=DTYPE)
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

            # Initialize momentum buffer if needed
            if p not in self.state:
                self.state[p] = {
                    "v": xp.zeros_like(p.data, dtype=DTYPE),  # momentum buffer
                }

            state = self.state[p]

            # Update momentum
            state["v"] = self.beta * state["v"] + (1 - self.beta) * grad

            # Parameter update
            p.data -= lr * state["v"]