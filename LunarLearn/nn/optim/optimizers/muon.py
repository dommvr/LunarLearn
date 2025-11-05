import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Muon(BaseOptimizer):
    """
    Muon optimizer with autograd support.

    This optimizer maintains momentum and an exponential moving average of the
    squared gradients (as a diagonal Hessian approximation). It updates parameters
    using a momentum-over-Hessian rule with damping to stabilize training.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.001.
        beta1 (float, optional): 
            Exponential decay rate for momentum. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for Hessian approximation. Default is 0.999.
        damping (float, optional): 
            Stabilizer added to denominator to avoid division by zero. Default is 1e-5.
        epsilon (float, optional): 
            Numerical stability constant for bias correction. Default is 1e-8.

    Attributes:
        state (dict): 
            Per-parameter state storing momentum and Hessian approximations.
        t (int): 
            Global timestep used for bias correction.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over parameters. Supports both raw 
            Tensors and dictionaries with {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, damping=1e-5, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.damping = xp.array(damping, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.t = 0
        self.state = {}

    def step(self, params):
        self.t += 1

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
                    "m": xp.zeros_like(p.data, dtype=DTYPE),   # momentum
                    "h": xp.zeros_like(p.data, dtype=DTYPE),   # Hessian approx
                }

            state = self.state[p]

            # Update momentum
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            # Update Hessian approximation (EMA of squared gradients)
            state["h"] = self.beta2 * state["h"] + (1 - self.beta2) * (grad * grad)

            # Bias correction
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            h_hat = state["h"] / (1 - xp.power(self.beta2, self.t))

            # Muon update rule
            step = m_hat / (xp.abs(h_hat) + self.damping)

            # Update parameter
            p.data -= lr * step