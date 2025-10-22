import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Adan(BaseOptimizer):
    """
    Adan optimizer with autograd support.

    Adan extends Adam by incorporating gradient differences and
    second-order information for more stable convergence, especially
    in large-batch training. It maintains momentum, difference
    momentum, and RMS statistics.

    Args:
        learning_rate (float, optional): Global learning rate. Default is 1e-3.
        beta1 (float, optional): Exponential decay for first moment. Default is 0.98.
        beta2 (float, optional): Exponential decay for gradient differences. Default is 0.92.
        beta3 (float, optional): Exponential decay for second-order moments. Default is 0.99.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-8.
        weight_decay (float, optional): Weight decay factor. Default is 0.0.

    Attributes:
        state (dict): Per-parameter optimizer state containing:
            - "m": First moment estimate.
            - "d": Gradient difference momentum.
            - "s": Second-order RMS statistics.
            - "prev_grad": Previous gradient for difference computation.
        t (int): Step counter for bias correction.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step on the given parameters.
        zero_grad(params: List[Tensor]):
            Reset gradients of the given parameters to None.
    """
    def __init__(self, learning_rate=1e-3, beta1=0.98, beta2=0.92,
                 beta3=0.99, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.beta3 = xp.array(beta3, dtype=DTYPE)
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
                    "m": xp.zeros_like(p.data, dtype=DTYPE),      # first moment
                    "d": xp.zeros_like(p.data, dtype=DTYPE),      # diff momentum
                    "s": xp.zeros_like(p.data, dtype=DTYPE),      # RMS
                    "prev_grad": xp.zeros_like(p.data, dtype=DTYPE),  # previous gradient
                }

            state = self.state[p]

            # Gradient difference
            delta_g = grad - state["prev_grad"]

            # Update moments
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["d"] = self.beta2 * state["d"] + (1 - self.beta2) * delta_g
            state["s"] = self.beta3 * state["s"] + (1 - self.beta3) * (grad * grad)

            # Bias correction
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            d_hat = state["d"] / (1 - xp.power(self.beta2, self.t))
            s_hat = state["s"] / (1 - xp.power(self.beta3, self.t))

            # Update rule
            step = (m_hat + self.beta2 * d_hat) / (xp.sqrt(s_hat) + self.epsilon)

            # Apply weight decay
            if self.weight_decay > 0:
                step += self.weight_decay * p.data

            p.data -= lr * step

            # Save current grad for next step
            state["prev_grad"] = grad