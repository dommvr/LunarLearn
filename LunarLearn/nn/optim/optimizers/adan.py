import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Adan(BaseOptimizer):
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
                    "d": xp.zeros_like(data, dtype=DTYPE),
                    "s": xp.zeros_like(data, dtype=DTYPE),
                    "prev_grad": xp.zeros_like(data, dtype=DTYPE),
                }
            state = self.state[param]
            delta_g = grad - state["prev_grad"]
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["d"] = self.beta2 * state["d"] + (1 - self.beta2) * delta_g
            state["s"] = self.beta3 * state["s"] + (1 - self.beta3) * (grad * grad)
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            d_hat = state["d"] / (1 - xp.power(self.beta2, self.t))
            s_hat = state["s"] / (1 - xp.power(self.beta3, self.t))
            step = (m_hat + self.beta2 * d_hat) / (xp.sqrt(s_hat) + self.epsilon)
            if self.weight_decay > 0:
                step += self.weight_decay * data
            lr = self._get_lr(param, layer)
            data -= lr * step
            state["prev_grad"] = grad
            self._apply_weight_decay(param, layer, lr)