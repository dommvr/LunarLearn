import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaDelta(BaseOptimizer):
    """
    AdaDelta optimizer with autograd support.

    AdaDelta is an adaptive learning rate method that eliminates the need to 
    manually set a learning rate. Instead, it dynamically adapts updates using 
    a moving window of squared gradients and squared updates.

    Args:
        rho (float, optional): Decay rate for the moving averages of squared 
            gradients and updates. Default is 0.95.
        epsilon (float, optional): Small constant to prevent division by zero. 
            Default is 1e-8.

    Attributes:
        rho (float): Decay rate for moving averages.
        epsilon (float): Numerical stability constant.
        state (dict): Per-parameter optimizer state, storing accumulated squared 
            gradients and updates.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over the given parameters.
        zero_grad(params: List[Tensor]):
            Reset gradients of all given parameters to None.
    """
    def __init__(self, rho=0.95, epsilon=1e-8):
        super().__init__(learning_rate=0.0)  # AdaDelta does not use a fixed LR
        self.rho = xp.array(rho, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.state = {}

    def step(self, params):
        for param_desc in params:
            # --- Extract param & layer if available ---
            if isinstance(param_desc, dict):
                p = param_desc["param"]
                layer = param_desc.get("layer", None)
            else:
                p = param_desc
                layer = None

            if not isinstance(p, Tensor) or not p.requires_grad:
                continue
            if p.grad is None:
                continue

            grad = p.grad

            # Initialize state if needed
            if p not in self.state:
                self.state[p] = {
                    "Eg2": xp.zeros_like(p.data, dtype=DTYPE),   # running avg of grad^2
                    "Edx2": xp.zeros_like(p.data, dtype=DTYPE),  # running avg of update^2
                }

            state = self.state[p]

            # Accumulate gradient squared
            state["Eg2"] = self.rho * state["Eg2"] + (1 - self.rho) * (grad * grad)

            # Compute update step
            update = - (xp.sqrt(state["Edx2"] + self.epsilon) /
                        xp.sqrt(state["Eg2"] + self.epsilon)) * grad

            # Accumulate updates squared
            state["Edx2"] = self.rho * state["Edx2"] + (1 - self.rho) * (update * update)

            # Apply update
            p.data += update
