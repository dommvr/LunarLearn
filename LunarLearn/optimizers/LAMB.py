import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class LAMB(BaseOptimizer):
    """
    Layer-wise Adaptive Moments (LAMB) optimizer with autograd support.

    LAMB extends Adam by introducing a "trust ratio" that scales updates 
    based on the norm of the parameters relative to the norm of the update. 
    It is particularly well-suited for large-batch training.

    Args:
        learning_rate (float, optional): Base learning rate. Default is 1e-3.
        beta1 (float, optional): Exponential decay for first moment. Default is 0.9.
        beta2 (float, optional): Exponential decay for second moment. Default is 0.999.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-6.

    Attributes:
        state (dict): Per-parameter optimizer state containing:
            - "m": First moment estimate.
            - "v": Second moment estimate.
        t (int): Step counter.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step on the given parameters.
        zero_grad(params: List[Tensor]):
            Reset gradients of the given parameters to None.
    """
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.t = 0
        self.state = {}

    def step(self, params):
        self.t += 1

        for param_desc in params:
            # Extract param & layer if available
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
                self.state[p] = {
                    "m": xp.zeros_like(p.data, dtype=DTYPE),  # first moment
                    "v": xp.zeros_like(p.data, dtype=DTYPE),  # second moment
                }

            state = self.state[p]

            # Update biased moments
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # Bias correction
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))

            # Compute Adam-style update
            update = m_hat / (xp.sqrt(v_hat) + self.epsilon)

            # Compute trust ratio
            w_norm = xp.linalg.norm(p.data)
            u_norm = xp.linalg.norm(update)
            trust_ratio = xp.where(w_norm > 0, w_norm / (u_norm + self.epsilon), 1.0)

            # Apply update
            p.data -= lr * trust_ratio * update