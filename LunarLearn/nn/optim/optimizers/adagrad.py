import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaGrad(BaseOptimizer):
    """
    AdaGrad optimizer with autograd support.

    AdaGrad adapts the learning rate for each parameter individually by
    scaling it inversely proportional to the square root of the sum of
    historical squared gradients. This allows frequent features to get
    smaller updates, and infrequent features to get larger updates.

    Args:
        learning_rate (float, optional): Global learning rate. Default is 1e-3.
        epsilon (float, optional): Small constant to avoid division by zero. Default is 1e-8.

    Attributes:
        state (dict): Per-parameter optimizer state containing:
            - "G": Accumulated squared gradients.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step on the given parameters.
        zero_grad(params: List[Tensor]):
            Reset gradients of the given parameters to None.
    """
    def __init__(self, learning_rate=1e-3, epsilon=1e-8):
        super().__init__(learning_rate)
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
            lr = self._get_lr(param_desc)

            # Initialize state if needed
            if p not in self.state:
                self.state[p] = {"G": xp.zeros_like(p.data, dtype=DTYPE)}

            state = self.state[p]

            # Accumulate squared gradients
            state["G"] += grad * grad

            # Compute update
            update = grad / (xp.sqrt(state["G"]) + self.epsilon)

            # Apply update
            p.data -= lr * update