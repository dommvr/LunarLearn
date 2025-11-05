import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaBound(BaseOptimizer):
    """
    AdaBound optimizer with autograd support.

    This optimizer extends Adam by dynamically bounding the learning rate 
    with lower and upper limits that tighten as training progresses. It 
    combines the fast convergence of Adam with the stability of SGD.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.001.
        final_lr (float, optional): 
            Final (SGD-like) learning rate towards which step sizes are bounded. Default is 0.1.
        beta1 (float, optional): 
            Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for the second moment estimates. Default is 0.999.
        gamma (float, optional): 
            Convergence speed of the bound functions. Default is 1e-3.
        epsilon (float, optional): 
            Small constant to avoid division by zero. Default is 1e-8.

    Attributes:
        state (dict): 
            Per-parameter state storing first and second moment estimates.
        t (int): 
            Global timestep, used for bias correction of moments.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Tensors and dictionaries containing {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=1e-3, final_lr=0.1, beta1=0.9, beta2=0.999,
                 gamma=1e-3, epsilon=1e-8):
        super().__init__(learning_rate)
        self.final_lr = xp.array(final_lr, dtype=DTYPE)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.gamma = xp.array(gamma, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.t = 0
        self.state = {}  # store moments per parameter

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

            # Update biased first and second moments
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # Bias correction
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))

            # Compute Adam step size
            step_size = lr / (xp.sqrt(v_hat) + self.epsilon)

            # Dynamic bounds on learning rate
            lower_bound = self.final_lr * (1 - 1 / (self.gamma * self.t + 1))
            upper_bound = self.final_lr * (1 + 1 / (self.gamma * self.t))

            step_size = xp.clip(step_size, lower_bound, upper_bound)

            # Update parameter
            p.data -= step_size * m_hat
