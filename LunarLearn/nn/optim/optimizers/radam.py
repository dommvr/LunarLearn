import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class RAdam(BaseOptimizer):
    """
    Rectified Adam (RAdam) optimizer with autograd support.

    This optimizer modifies Adam by introducing a rectification term that 
    stabilizes variance in the early training steps. When variance estimates 
    are reliable (rho_t > 4), it applies rectified Adam updates; otherwise, 
    it falls back to SGD with momentum-like behavior.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.001.
        beta1 (float, optional): 
            Exponential decay rate for first moment. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for second moment. Default is 0.999.
        epsilon (float, optional): 
            Small constant to avoid division by zero. Default is 1e-8.

    Attributes:
        state (dict): 
            Per-parameter state storing first and second moment estimates.
        t (int): 
            Global timestep, used for bias correction and rectification.
        rho_inf (float): 
            Maximum value of rho_t used in rectification.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.t = 0

        # Precompute constant for rectification
        self.rho_inf = 2.0 / (1.0 - self.beta2) - 1.0
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

            # --- First moment update ---
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            # Bias correction for first moment
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))

            # --- Second moment update ---
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # Compute rectification term
            beta2_t = xp.power(self.beta2, self.t)
            rho_t = self.rho_inf - 2.0 * self.t * beta2_t / (1.0 - beta2_t)

            if rho_t > 4:  # Variance rectification is reliable
                # Variance rectification factor
                r_t = xp.sqrt(
                    ((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf)
                    / ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t)
                )

                # Bias correction for second moment
                v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))

                # Update with rectification
                p.data -= lr * r_t * m_hat / (xp.sqrt(v_hat) + self.epsilon)

            else:
                # Too early -> fall back to SGD with momentum
                p.data -= lr * m_hat