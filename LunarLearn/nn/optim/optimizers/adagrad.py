import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaGrad(BaseOptimizer):
    def __init__(self, learning_rate=1e-3, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
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
                self.state[param] = {"G": xp.zeros_like(data, dtype=DTYPE)}
            state = self.state[param]
            state["G"] += grad * grad
            lr = self._get_lr(param, layer)
            update = grad / (xp.sqrt(state["G"]) + self.epsilon)
            data -= lr * update
            self._apply_weight_decay(param, layer, lr)