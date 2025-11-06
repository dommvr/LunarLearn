import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaFactor(BaseOptimizer):
    """
    Adafactor optimizer with autograd support.

    Adafactor is a memory-efficient adaptive optimizer that factorizes
    the second-moment estimates for large 2D parameters, while falling
    back to full storage for 1D parameters (like biases). It supports
    relative step sizes that adapt learning rate to parameter scale.

    Args:
        learning_rate (float, optional): Base learning rate. Default is 1e-3.
        beta2 (float, optional): Decay rate for second-moment estimates. Default is 0.999.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-8.
        relative_step (bool, optional): If True, scale step size relative to parameter norm. Default is True.

    Attributes:
        state (dict): Per-parameter optimizer state.
            For 2D parameters: {"row_avg", "col_avg"}  
            For 1D parameters: {"v"}  

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over the given parameters.
        zero_grad(params: List[Tensor]):
            Reset gradients of all given parameters to None.
    """
    def __init__(self, learning_rate=1e-3, beta2=0.999, epsilon=1e-8, relative_step=True):
        super().__init__(learning_rate)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.relative_step = relative_step
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

            # --- Relative step scaling ---
            if self.relative_step:
                rms = xp.sqrt(xp.mean(p.data * p.data))
                lr = self.learning_rate0 * xp.maximum(rms, 1e-6)

            # Initialize state if needed
            if p not in self.state:
                if p.data.ndim == 2:  # Factorized for matrices
                    self.state[p] = {
                        "row_avg": xp.zeros(p.data.shape[0], dtype=DTYPE),
                        "col_avg": xp.zeros(p.data.shape[1], dtype=DTYPE),
                    }
                else:  # Full storage for vectors/biases
                    self.state[p] = {"v": xp.zeros_like(p.data, dtype=DTYPE)}

            state = self.state[p]

            # --- Update second moment estimates ---
            if p.data.ndim == 2:
                g2 = grad * grad
                state["row_avg"] = self.beta2 * state["row_avg"] + (1 - self.beta2) * xp.mean(g2, axis=1)
                state["col_avg"] = self.beta2 * state["col_avg"] + (1 - self.beta2) * xp.mean(g2, axis=0)

                # Reconstruct factorized variance
                v_hat = xp.outer(state["row_avg"], state["col_avg"]) / xp.mean(state["row_avg"])
            else:
                state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)
                v_hat = state["v"]

            # --- Compute parameter update ---
            update = grad / (xp.sqrt(v_hat) + self.epsilon)

            # Apply update
            p.data -= lr * update
