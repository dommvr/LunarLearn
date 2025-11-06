import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaBound(BaseOptimizer):
    """
    AdaBound optimizer with autograd support.

    This optimizer implements the AdaBound algorithm, which applies dynamic bounds
    on learning rates to combine the benefits of adaptive methods like Adam with
    the generalization of SGD by transitioning to fixed learning rates over time.
    Args:
        learning_rate (float, optional):
            Initial learning rate. Default is 1e-3.
        final_lr (float, optional):
            Final learning rate for bounding. Default is 0.1.
        beta1 (float, optional):
            Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 (float, optional):
            Exponential decay rate for the second moment estimates. Default is 0.999.
        gamma (float, optional):
            Convergence speed of the bound functions. Default is 1e-3.
        epsilon (float, optional):
            Small constant for numerical stability. Default is 1e-8.
    Attributes:
        state (dict):
            Per-parameter state storing first and second moment estimates.
        t (int):
            Global timestep, used for bias correction of moments.
    Methods:
        step(params: List[Union[Parameter, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Parameters and dictionaries containing {"param": Parameter, "layer": Layer}.
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
        self.state = {} # store moments per parameter
    def step(self, params):
        self.t += 1
        for param, layer, custom_optim in self._iter_params(params):
            # If param has its own optimizer → delegate and skip
            if custom_optim is not None and custom_optim is not self:
                custom_optim.step([param]) # Step only that param
                continue
            # Otherwise, use *this* optimizer to update it
            grad = param.grad
            data = param.data
            if param not in self.state:
                self.state[param] = {
                    "m": xp.zeros_like(data, dtype=DTYPE),
                    "v": xp.zeros_like(data, dtype=DTYPE),
                }
            state = self.state[param]
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))
            lr = self._get_lr(param, layer)
            step_size = lr / (xp.sqrt(v_hat) + self.epsilon)
            # Dynamic bounds on learning rate
            lower_bound = self.final_lr * (1 - 1 / (self.gamma * self.t + 1))
            upper_bound = self.final_lr * (1 + 1 / (self.gamma * self.t))
            step_size = xp.clip(step_size, lower_bound, upper_bound)
            data -= step_size * m_hat
            self._apply_weight_decay(param, layer, lr)
