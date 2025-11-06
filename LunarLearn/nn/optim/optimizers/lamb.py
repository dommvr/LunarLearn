import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class LAMB(BaseOptimizer):
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6):
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
            update = m_hat / (xp.sqrt(v_hat) + self.epsilon)
            w_norm = xp.linalg.norm(data)
            u_norm = xp.linalg.norm(update)
            trust_ratio = xp.where(w_norm > 0, w_norm / (u_norm + self.epsilon), 1.0)
            lr = self._get_lr(param, layer)
            data -= lr * trust_ratio * update
            self._apply_weight_decay(param, layer, lr)