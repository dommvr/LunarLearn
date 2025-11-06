import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Ranger(BaseOptimizer):
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
                    "slow": data.copy(),
                }
            state = self.state[param]
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)
            beta2_t = xp.power(self.beta2, self.t)
            rho_t = self.rho_inf - 2.0 * self.t * beta2_t / (1.0 - beta2_t)
            lr = self._get_lr(param, layer)
            if rho_t > 4:
                r_t = xp.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf) /
                              ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t))
                v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))
                update = r_t * m_hat / (xp.sqrt(v_hat) + self.epsilon)
            else:
                update = m_hat
            data -= lr * update
            # Lookahead update
            if self.step_counter % self.k == 0:
                state["slow"] += self.alpha * (data - state["slow"])
                data = state["slow"].copy()
            self._apply_weight_decay(param, layer, lr)