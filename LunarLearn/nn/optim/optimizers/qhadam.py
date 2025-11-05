import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class QHAdam(BaseOptimizer):
    """
    Quasi-Hyperbolic Adam optimizer with autograd support.

    This optimizer extends Adam by introducing hyperparameters (nu1, nu2) 
    that interpolate between the current gradient and the exponential 
    moving averages of the first and second moments. This can improve 
    stability and speed of convergence.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 1e-3.
        beta1 (float, optional): 
            Exponential decay rate for first moment. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for second moment. Default is 0.999.
        nu1 (float, optional): 
            Interpolation factor for first moment (between current grad and EMA). Default is 0.7.
        nu2 (float, optional): 
            Interpolation factor for second moment (between grad^2 and EMA). Default is 1.0.
        epsilon (float, optional): 
            Small constant to avoid division by zero. Default is 1e-8.

    Attributes:
        state (dict): 
            Per-parameter state storing EMA of first and second moments.
        t (int): 
            Global timestep for bias correction.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. 
            Supports raw Tensors or dicts {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999,
                 nu1=0.7, nu2=1.0, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.nu1 = xp.array(nu1, dtype=DTYPE)
        self.nu2 = xp.array(nu2, dtype=DTYPE)
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
                    "m": xp.zeros_like(p.data, dtype=DTYPE),  # EMA of gradient
                    "v": xp.zeros_like(p.data, dtype=DTYPE),  # EMA of squared gradient
                }

            state = self.state[p]

            # Update exponential moving averages
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # Quasi-hyperbolic interpolation
            m_qh = (1 - self.nu1) * grad + self.nu1 * state["m"]
            v_qh = (1 - self.nu2) * (grad * grad) + self.nu2 * state["v"]

            # Update parameters
            p.data -= lr * m_qh / (xp.sqrt(v_qh) + self.epsilon)