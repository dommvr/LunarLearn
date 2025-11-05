import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class AdaNorm(BaseOptimizer):
    """
    AdaNorm / AdaNorm+ optimizer with autograd support.

    AdaNorm normalizes gradients by their RMS value to stabilize training. 
    AdaNorm+ extends it by applying EMA smoothing on the RMS.

    Args:
        learning_rate (float, optional): Base learning rate. Default is 1e-3.
        beta1 (float, optional): Exponential decay for first moment. Default is 0.9.
        beta2 (float, optional): Exponential decay for second moment. Default is 0.999.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-8.
        use_plus (bool, optional): If True, uses AdaNorm+ (EMA of RMS). Default is False.
        rms_beta (float, optional): Decay factor for AdaNorm+ RMS smoothing. Default is 0.95.
        bias_correction (bool, optional): Whether to apply bias correction to moments. Default is True.

    Attributes:
        state (dict): Per-parameter optimizer state containing:
            - "m": First moment estimate.
            - "v": Second moment estimate.
            - "ema_rms": Smoothed RMS (for AdaNorm+ only).
        t (int): Step counter.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step on the given parameters.
        zero_grad(params: List[Tensor]):
            Reset gradients of the given parameters to None.
    """
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_plus=False, rms_beta=0.95, bias_correction=True):
        super().__init__(learning_rate)
        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.use_plus = use_plus
        self.rms_beta = xp.array(rms_beta, dtype=DTYPE)
        self.bias_correction = bias_correction
        self.t = 0
        self.state = {}

    def step(self, params):
        self.t += 1

        for param_desc in params:
            # --- Extract parameter & layer if available ---
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

            # Lazy initialization of optimizer state
            if p not in self.state:
                self.state[p] = {
                    "m": xp.zeros_like(p.data, dtype=DTYPE),    # first moment
                    "v": xp.zeros_like(p.data, dtype=DTYPE),    # second moment
                    "ema_rms": xp.array(0.0, dtype=DTYPE) if self.use_plus else None
                }

            state = self.state[p]

            # Gradient normalization 
            if self.use_plus:
                grad_rms = xp.sqrt(xp.mean(grad ** 2)) + self.epsilon
                state["ema_rms"] = self.rms_beta * state["ema_rms"] + (1 - self.rms_beta) * grad_rms
                norm_factor = state["ema_rms"]
            else:
                norm_factor = xp.sqrt(xp.mean(grad ** 2)) + self.epsilon

            g_normed = grad / norm_factor

            # Update biased moment estimates
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * g_normed
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

            # Bias correction (optional)
            if self.bias_correction:
                m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))
                v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))
            else:
                m_hat = state["m"]
                v_hat = state["v"]

            # Parameter update
            denom = xp.sqrt(v_hat) + self.epsilon
            update = m_hat / denom
            p.data -= lr * update
