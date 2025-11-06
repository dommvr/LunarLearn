import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class RMSProp(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8, weight_decay=0.0):
        super().__init__(learning_rate)
        self.beta = xp.array(beta, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.weight_decay = xp.array(weight_decay, dtype=DTYPE)
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
            state["v"] = self.beta * state["v"] + (1 - self.beta) * (grad * grad)
            lr = self._get_lr(param, layer)
            update = grad / (xp.sqrt(state["v"]) + self.epsilon)
            if self.weight_decay > 0:
                update += self.weight_decay * data
            data -= lr * update
            self._apply_weight_decay(param, layer, lr)