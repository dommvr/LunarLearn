import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Ranger(BaseOptimizer):
    """
    Ranger optimizer (RAdam + Lookahead).

    This optimizer combines variance rectification from RAdam with 
    the Lookahead mechanism. RAdam stabilizes variance in early training, 
    while Lookahead improves convergence stability by interpolating 
    fast weights with slow weights.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.001.
        beta1 (float, optional): 
            Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for the second moment estimates. Default is 0.999.
        epsilon (float, optional): 
            Small constant for numerical stability. Default is 1e-8.
        k (int, optional): 
            Number of steps before Lookahead interpolation. Default is 6.
        alpha (float, optional): 
            Interpolation factor between fast and slow weights. Default is 0.5.

    Attributes:
        state (dict): 
            Per-parameter state storing moments and lookahead weights.
        t (int): 
            Global timestep for bias correction and rectification.
        rho_inf (float): 
            Maximum value of rho_t used in rectification.
        step_counter (int): 
            Counter to track when to perform Lookahead updates.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, k=6, alpha=0.5):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)

        self.k = k
        self.alpha = alpha

        self.t = 0
        self.step_counter = 0
        self.rho_inf = 2.0 / (1.0 - self.beta2) - 1.0
        self.state = {}

    def step(self, params):
        self.t += 1
        self.step_counter += 1

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
                    "slow": p.data.copy(),                   # lookahead slow weights
                }

            state = self.state[p]

            # RAdam update
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))

            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            beta2_t = xp.power(self.beta2, self.t)
            rho_t = self.rho_inf - 2.0 * self.t * beta2_t / (1.0 - beta2_t)

            if rho_t > 4:
                # Rectified variance
                r_t = xp.sqrt(
                    ((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf)
                    / ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t)
                )
                v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))
                update = r_t * m_hat / (xp.sqrt(v_hat) + self.epsilon)
            else:
                # Too early → SGD with momentum fallback
                update = m_hat

            p.data -= lr * update

            # Lookahead update
            if self.step_counter % self.k == 0:
                state["slow"] += self.alpha * (p.data - state["slow"])
                p.data = state["slow"].copy()