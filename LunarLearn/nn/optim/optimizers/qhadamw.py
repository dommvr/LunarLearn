import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class QHAdamW(BaseOptimizer):
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
            m_qh = (1 - self.nu1) * grad + self.nu1 * state["m"]
            v_qh = (1 - self.nu2) * (grad * grad) + self.nu2 * state["v"]
            lr = self._get_lr(param, layer)
            # Decoupled weight decay
            if self.weight_decay > 0:
                data -= lr * self.weight_decay * data
            # Parameter update
            data -= lr * m_qh / (xp.sqrt(v_qh) + self.epsilon)
            self._apply_weight_decay(param, layer, lr)