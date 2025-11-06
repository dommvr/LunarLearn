import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class Lion(BaseOptimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.state = {}

    def step(self, params):
        for param, layer, custom_optim in self._iter_params(params):
            # If param has its own optimizer → delegate and skip
            if custom_optim is not None and custom_optim is not self:
                custom_optim.step([param])
                continue
            # Otherwise, use *this* optimizer to update it
            grad = param.grad
            data = param.data
            if param not in self.state:
                self.state[param] = {"m": xp.zeros_like(data, dtype=DTYPE)}
            state = self.state[param]
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            lr = self._get_lr(param, layer)
            data -= lr * xp.sign(state["m"])
            state["m"] *= self.beta2
            self._apply_weight_decay(param, layer, lr)