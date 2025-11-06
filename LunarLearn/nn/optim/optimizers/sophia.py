import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Sophia(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.965, beta2=0.99,
                 rho=0.04, epsilon=1e-8, mode="G"):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.rho = xp.array(rho, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.mode = mode.upper()
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

            if param not in self.state:
                state = {"m": xp.zeros_like(data, dtype=DTYPE)}
                if self.mode == "G":
                    state["h"] = xp.zeros_like(data, dtype=DTYPE)
                elif self.mode == "H":
                    state["h"] = xp.array(0.0, dtype=DTYPE)
                self.state[param] = state
            else:
                state = self.state[param]

            # Update momentum
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            if self.mode == "G":
                # Per-weight diagonal Hessian estimate
                state["h"] = self.beta2 * state["h"] + (1 - self.beta2) * (grad * grad)
                h_clipped = xp.minimum(state["h"], self.rho)
                data -= lr * state["m"] / (h_clipped + self.epsilon)
            elif self.mode == "H":
                # Per-layer scalar trace estimate
                h_trace = xp.mean(grad * grad)
                state["h"] = self.beta2 * state["h"] + (1 - self.beta2) * h_trace
                h_clipped = xp.minimum(state["h"], self.rho)
                data -= lr * state["m"] / (h_clipped + self.epsilon)

            self._apply_weight_decay(param, layer, lr)