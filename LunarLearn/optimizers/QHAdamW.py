import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class QHAdamW(BaseOptimizer):
    """
    Quasi-Hyperbolic AdamW optimizer with autograd support.

    This optimizer extends AdamW by introducing quasi-hyperbolic terms (nu1, nu2) 
    that interpolate between the raw gradients and their exponential moving averages. 
    It also applies decoupled weight decay (AdamW style).

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 1e-3.
        beta1 (float, optional): 
            Exponential decay rate for first moment. Default is 0.95.
        beta2 (float, optional): 
            Exponential decay rate for second moment. Default is 0.999.
        nu1 (float, optional): 
            Interpolation factor for first moment (gradient vs EMA). Default is 0.7.
        nu2 (float, optional): 
            Interpolation factor for second moment (grad^2 vs EMA). Default is 1.0.
        epsilon (float, optional): 
            Small constant to avoid division by zero. Default is 1e-8.
        weight_decay (float, optional): 
            Decoupled weight decay factor. Default is 0.01.

    Attributes:
        state (dict): 
            Per-parameter state storing EMA of first and second moments.
        t (int): 
            Global timestep (not used for bias correction here, 
            but useful if extended later).

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. 
            Supports raw Tensors or dicts {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=1e-3, beta1=0.95, beta2=0.999,
                 nu1=0.7, nu2=1.0, epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.nu1 = xp.array(nu1, dtype=DTYPE)
        self.nu2 = xp.array(nu2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.weight_decay = xp.array(weight_decay, dtype=DTYPE)
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

            # EMA updates (Adam-style)
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # QH interpolation
            m_qh = (1 - self.nu1) * grad + self.nu1 * state["m"]
            v_qh = (1 - self.nu2) * (grad * grad) + self.nu2 * state["v"]

            # Decoupled weight decay (AdamW style)
            if self.weight_decay > 0:
                p.data -= lr * self.weight_decay * p.data

            # Parameter update
            p.data -= lr * m_qh / (xp.sqrt(v_qh) + self.epsilon)