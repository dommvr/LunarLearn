import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Nadam(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.t = 0
        self.state = {}

    def step(self, params):
        self.t += 1
        for param, layer, custom_optim in self._iter_params(params):
            # If param has its own optimizer → delegate and skip
            if custom_optim is not None and custom_optim is not self:
                custom_optim.step([param])
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
            # Nesterov lookahead
            nesterov_m = self.beta1 * m_hat + (1 - self.beta1) * grad / (1 - xp.power(self.beta1, self.t))
            lr = self._get_lr(param, layer)
            data -= lr * nesterov_m / (xp.sqrt(v_hat) + self.epsilon)
            self._apply_weight_decay(param, layer, lr)