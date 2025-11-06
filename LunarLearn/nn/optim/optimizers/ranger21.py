import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Ranger21(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.01, k=6, alpha=0.5, ema_decay=0.999,
                 use_gc=True, use_radam=True, use_norm_loss=True, use_ema=True,
                 use_clip=True, clip_value=1.0, clip_norm=None):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.weight_decay = xp.array(weight_decay, dtype=DTYPE)
        self.alpha = xp.array(alpha, dtype=DTYPE)
        self.ema_decay = xp.array(ema_decay, dtype=DTYPE)
        self.use_gc = use_gc
        self.use_radam = use_radam
        self.use_norm_loss = use_norm_loss
        self.use_ema = use_ema
        self.use_clip = use_clip
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.k = k
        self.t = 0
        self.step_counter = 0
        self.rho_inf = 2.0 / (1.0 - self.beta2) - 1.0
        self.state = {}

    def _centralize_grad(self, g):
        if g.ndim > 1:
            return g - g.mean(axis=tuple(range(1, g.ndim)), keepdims=True)
        return g

    def _clip_grad(self, g):
        if self.clip_value is not None:
            g = xp.clip(g, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            norm = xp.linalg.norm(g)
            if norm > self.clip_norm:
                g = g * (self.clip_norm / (norm + 1e-6))
        return g

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
            if self.use_gc:
                grad = self._centralize_grad(grad)
            if self.use_clip:
                grad = self._clip_grad(grad)
            lr = self._get_lr(param, layer)
            if param not in self.state:
                self.state[param] = {
                    "m": xp.zeros_like(data, dtype=DTYPE),
                    "v": xp.zeros_like(data, dtype=DTYPE),
                    "slow": data.copy(),
                    "ema": data.copy() if self.use_ema else None,
                }
            state = self.state[param]
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))
            r_t = 1.0
            if self.use_radam:
                beta2_t = xp.power(self.beta2, self.t)
                rho_t = self.rho_inf - 2.0 * self.t * beta2_t / (1.0 - beta2_t)
                if rho_t > 4:
                    r_t = xp.sqrt(((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf) /
                                  ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t))
            update = r_t * m_hat / (xp.sqrt(v_hat) + self.epsilon)
            if self.use_norm_loss:
                w_norm = xp.linalg.norm(data)
                if w_norm > 0:
                    update /= w_norm
            if self.weight_decay > 0:
                data -= lr * self.weight_decay * data
            data -= lr * update
            if self.step_counter % self.k == 0:
                state["slow"] += self.alpha * (data - state["slow"])
                data = state["slow"].copy()
            if self.use_ema:
                state["ema"] = self.ema_decay * state["ema"] + (1 - self.ema_decay) * data
            self._apply_weight_decay(param, layer, lr)

    def swap_weights_with_ema(self, params):
        if not self.use_ema:
            return
        for param, layer, custom_optim in self._iter_params(params):
            if param not in self.state or self.state[param]["ema"] is None:
                continue
            param.data, self.state[param]["ema"] = self.state[param]["ema"], param.data