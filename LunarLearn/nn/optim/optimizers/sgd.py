import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class SGD(BaseOptimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def step(self, params):
        for param, layer, custom_optim in self._iter_params(params):
            # If param has its own optimizer → delegate and skip
            if custom_optim is not None and custom_optim is not self:
                custom_optim.step([param])
                continue
            # Otherwise, use *this* optimizer to update it
            grad = param.grad
            data = param.data
            lr = self._get_lr(param, layer)
            data -= lr * grad
            self._apply_weight_decay(param, layer, lr)
