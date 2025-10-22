import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdamW(BaseOptimizer):
    """
    AdamW optimizer with autograd support.

    AdamW is a variant of Adam that decouples weight decay from the
    gradient-based updates. It combines adaptive learning rates with
    decoupled L2 regularization for improved generalization.

    Args:
        learning_rate (float, optional): Global learning rate. Default is 1e-3.
        beta1 (float, optional): Exponential decay rate for first moment estimates. Default is 0.9.
        beta2 (float, optional): Exponential decay rate for second moment estimates. Default is 0.999.
        epsilon (float, optional): Small constant to avoid division by zero. Default is 1e-8.
        weight_decay (float, optional): Decoupled weight decay factor. Default is 0.01.

    Attributes:
        state (dict): Per-parameter optimizer state containing:
            - "m": First moment estimate.
            - "v": Second moment estimate.
        t (int): Step counter for bias correction.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step on the given parameters.
        zero_grad(params: List[Tensor]):
            Reset gradients of the given parameters to None.
    """
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.weight_decay = xp.array(weight_decay, dtype=DTYPE)
        self.t = 0
        self.state = {}

    def step(self, params):
        self.t += 1

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
                self.state[p] = {
                    "m": xp.zeros_like(p.data, dtype=DTYPE),  # first moment
                    "v": xp.zeros_like(p.data, dtype=DTYPE),  # second moment
                }

            state = self.state[p]

            # Update biased first moment
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            # Update biased second moment
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # Bias correction
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))

            # Parameter update
            p.data -= lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)

            # Decoupled weight decay
            if self.weight_decay > 0:
                p.data -= lr * self.weight_decay * p.data
