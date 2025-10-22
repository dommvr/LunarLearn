import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Nadam(BaseOptimizer):
    """
    Nadam optimizer with autograd support.

    This optimizer extends Adam by incorporating Nesterov momentum into the 
    update rule. It adapts learning rates per parameter using first and second 
    moments while providing a lookahead gradient adjustment.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.001.
        beta1 (float, optional): 
            Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for the second moment estimates. Default is 0.999.
        epsilon (float, optional): 
            Small constant to avoid division by zero. Default is 1e-8.

    Attributes:
        state (dict): 
            Per-parameter state storing first and second moment estimates.
        t (int): 
            Global timestep, used for bias correction.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Tensors and dictionaries containing {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
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
                    "m": xp.zeros_like(p.data, dtype=DTYPE),  # first moment
                    "v": xp.zeros_like(p.data, dtype=DTYPE),  # second moment
                }

            state = self.state[p]

            # Update biased first moment
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            # Update biased second moment
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # Bias correction
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))

            # Nadam: add Nesterov lookahead adjustment
            nesterov_m = self.beta1 * m_hat + (1 - self.beta1) * grad / (1 - xp.power(self.beta1, self.t))

            # Update parameters
            p.data -= lr * nesterov_m / (xp.sqrt(v_hat) + self.epsilon)