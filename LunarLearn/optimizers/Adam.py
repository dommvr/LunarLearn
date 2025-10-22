import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Adam(BaseOptimizer):
    """
    Adam optimizer with autograd support.

    This optimizer implements the Adam algorithm, which combines momentum 
    and adaptive learning rates for each parameter. It is widely used for 
    training deep learning models due to its fast convergence and robustness.

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.001.
        beta1 (float, optional): 
            Exponential decay rate for the first moment estimates. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for the second moment estimates. Default is 0.999.
        epsilon (float, optional): 
            Small constant to avoid division by zero. Default is 1e-8.

    Attributes:
        state (dict): 
            Per-parameter state storing first and second moment estimates.
        t (int): 
            Global timestep, used for bias correction of moments.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Tensors and dictionaries containing {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.t = 0
        self.state = {}  # store moments per parameter

    def step(self, params):
        self.t += 1

        for param, layer, custom_optim in self._iter_params(params):
            # If param has its own optimizer → delegate and skip
            if custom_optim is not None and custom_optim is not self:
                custom_optim.step([param])   # Step only that param
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

            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))

            lr = self._get_lr(param, layer)

            data -= lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)

            self._apply_weight_decay(param, layer, lr)