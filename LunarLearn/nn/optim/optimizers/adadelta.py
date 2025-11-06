import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class AdaDelta(BaseOptimizer):
    """
    AdaDelta optimizer with autograd support.
    AdaDelta is an adaptive learning rate method that eliminates the need to 
    manually set a learning rate. Instead, it dynamically adapts updates using 
    a moving window of squared gradients and squared updates.
    Args:
        learning_rate (float, optional):
            Learning rate scaling factor. Default is 1.0.
        rho (float, optional):
            Decay rate for the moving averages of squared gradients and updates. Default is 0.95.
        epsilon (float, optional):
            Small constant for numerical stability. Default is 1e-8.
    Attributes:
        state (dict):
            Per-parameter state storing accumulated squared gradients and updates.
    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Tensors and dictionaries containing {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-8):
        super().__init__(learning_rate)
        self.rho = xp.array(rho, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.state = {} # store moments per parameter

    def step(self, params):
        for param, layer, custom_optim in self._iter_params(params):
            # If param has its own optimizer → delegate and skip
            if custom_optim is not None and custom_optim is not self:
                custom_optim.step([param]) # Step only that param
                continue
            # Otherwise, use *this* optimizer to update it
            grad = param.grad
            data = param.data
            if param not in self.state:
                self.state[param] = {
                    "Eg2": xp.zeros_like(data, dtype=DTYPE),
                    "Edx2": xp.zeros_like(data, dtype=DTYPE),
                }
            state = self.state[param]
            state["Eg2"] = self.rho * state["Eg2"] + (1 - self.rho) * (grad * grad)
            rms_g = xp.sqrt(state["Eg2"] + self.epsilon)
            rms_dx = xp.sqrt(state["Edx2"] + self.epsilon)
            delta = (rms_dx / rms_g) * grad
            lr = self._get_lr(param, layer)
            update = -lr * delta
            data += update
            state["Edx2"] = self.rho * state["Edx2"] + (1 - self.rho) * (delta * delta)
            self._apply_weight_decay(param, layer, lr)
