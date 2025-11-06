import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class SGDMomentum(BaseOptimizer):
    def __init__(self, learning_rate=0.01, beta=0.9):
        super().__init__(learning_rate)
        self.beta = xp.array(beta, dtype=DTYPE)
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
                self.state[param] = {"v": xp.zeros_like(data, dtype=DTYPE)}
            state = self.state[param]
            state["v"] = self.beta * state["v"] + (1 - self.beta) * grad
            lr = self._get_lr(param, layer)
            data -= lr * state["v"]
            self._apply_weight_decay(param, layer, lr)