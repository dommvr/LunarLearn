import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class AdaNorm(BaseOptimizer):
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_plus=False, rms_beta=0.95, bias_correction=True):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.use_plus = use_plus
        self.rms_beta = xp.array(rms_beta, dtype=DTYPE)
        self.bias_correction = bias_correction
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
                    "ema_rms": xp.array(0.0, dtype=DTYPE) if self.use_plus else None
                }
            state = self.state[param]
            # Gradient normalization
            if self.use_plus:
                grad_rms = xp.sqrt(xp.mean(grad ** 2)) + self.epsilon
                state["ema_rms"] = self.rms_beta * state["ema_rms"] + (1 - self.rms_beta) * grad_rms
                norm_factor = state["ema_rms"]
            else:
                norm_factor = xp.sqrt(xp.mean(grad ** 2)) + self.epsilon
            g_normed = grad / norm_factor
            # Update moments
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * g_normed
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)
            # Bias correction
            if self.bias_correction:
                m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
                v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))
            else:
                m_hat = state["m"]
                v_hat = state["v"]
            # Apply update
            lr = self._get_lr(param, layer)
            update = m_hat / (xp.sqrt(v_hat) + self.epsilon)
            data -= lr * update
            self._apply_weight_decay(param, layer, lr)
