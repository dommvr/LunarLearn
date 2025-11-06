import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaFactor(BaseOptimizer):
    def __init__(self, learning_rate=1e-3, beta2=0.999, epsilon=1e-8, relative_step=True):
        super().__init__(learning_rate)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.relative_step = relative_step
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
            lr = self._get_lr(param, layer)

            # Relative step scaling
            if self.relative_step:
                rms = xp.sqrt(xp.mean(data * data))
                lr = self.learning_rate0 * xp.maximum(rms, 1e-6)

            # Initialize state
            if param not in self.state:
                if data.ndim == 2:
                    self.state[param] = {
                        "row_avg": xp.zeros(data.shape[0], dtype=DTYPE),
                        "col_avg": xp.zeros(data.shape[1], dtype=DTYPE),
                    }
                else:
                    self.state[param] = {"v": xp.zeros_like(data, dtype=DTYPE)}

            state = self.state[param]

            # Update second moment estimates
            if data.ndim == 2:
                g2 = grad * grad
                state["row_avg"] = self.beta2 * state["row_avg"] + (1 - self.beta2) * xp.mean(g2, axis=1)
                state["col_avg"] = self.beta2 * state["col_avg"] + (1 - self.beta2) * xp.mean(g2, axis=0)
                v_hat = xp.outer(state["row_avg"], state["col_avg"]) / xp.mean(state["row_avg"])
            else:
                state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)
                v_hat = state["v"]

            # Compute and apply update
            update = grad / (xp.sqrt(v_hat) + self.epsilon)
            data -= lr * update
            self._apply_weight_decay(param, layer, lr)
