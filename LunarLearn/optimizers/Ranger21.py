import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Ranger21(BaseOptimizer):
    """
    Ranger21 optimizer: AdamW + Lookahead + optional RAdam +
    Gradient Centralization + Norm Loss Scaling + EMA + Gradient Clipping.

    Combines several recent optimizer advances into a single algorithm.
    Supports:
      - AdamW base
      - RAdam rectification (optional)
      - Lookahead interpolation
      - Gradient Centralization (optional)
      - Norm Loss scaling (optional)
      - EMA tracking of weights (optional)
      - Gradient clipping (optional)

    Args:
        learning_rate (float, optional): 
            Initial learning rate. Default is 0.001.
        beta1 (float, optional): 
            Exponential decay rate for first moment. Default is 0.9.
        beta2 (float, optional): 
            Exponential decay rate for second moment. Default is 0.999.
        epsilon (float, optional): 
            Numerical stability constant. Default is 1e-8.
        weight_decay (float, optional): 
            Decoupled weight decay coefficient. Default is 0.01.
        k (int, optional): 
            Lookahead update frequency. Default is 6.
        alpha (float, optional): 
            Lookahead interpolation factor. Default is 0.5.
        ema_decay (float, optional): 
            EMA decay rate for weights. Default is 0.999.
        use_gc (bool, optional): 
            Whether to apply Gradient Centralization. Default is True.
        use_radam (bool, optional): 
            Whether to apply RAdam variance rectification. Default is True.
        use_norm_loss (bool, optional): 
            Whether to normalize weight updates by parameter norm. Default is True.
        use_ema (bool, optional): 
            Whether to track Exponential Moving Average of weights. Default is True.
        use_clip (bool, optional): 
            Whether to apply gradient clipping. Default is True.
        clip_value (float, optional): 
            Maximum absolute value for gradients. Default is 1.0.
        clip_norm (float, optional): 
            Maximum L2 norm for gradients. Default is None (disabled).
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.01, k=6, alpha=0.5, ema_decay=0.999,
                 use_gc=True, use_radam=True, use_norm_loss=True, use_ema=True,
                 use_clip=True, clip_value=1.0, clip_norm=None):

        super().__init__(learning_rate)

        self.beta1 = xp.array(beta1, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.weight_decay = xp.array(weight_decay, dtype=DTYPE)
        self.alpha = xp.array(alpha, dtype=DTYPE)
        self.ema_decay = xp.array(ema_decay, dtype=DTYPE)

        # Options
        self.use_gc = use_gc
        self.use_radam = use_radam
        self.use_norm_loss = use_norm_loss
        self.use_ema = use_ema
        self.use_clip = use_clip
        self.clip_value = clip_value
        self.clip_norm = clip_norm

        # Counters
        self.k = k
        self.t = 0
        self.step_counter = 0
        self.rho_inf = 2.0 / (1.0 - self.beta2) - 1.0

        self.state = {}

    def _centralize_grad(self, g):
        if g.ndim > 1:
            return g - g.mean(axis=tuple(range(1, g.ndim)), keepdims=True)
        return g

    def _clip_grad(self, g):
        """Apply gradient clipping (value or norm)."""
        if self.clip_value is not None:
            g = xp.clip(g, -self.clip_value, self.clip_value)
        if self.clip_norm is not None:
            norm = xp.linalg.norm(g)
            if norm > self.clip_norm:
                g = g * (self.clip_norm / (norm + 1e-6))
        return g

    def step(self, params):
        self.t += 1
        self.step_counter += 1

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

            g = p.grad
            if self.use_gc:
                g = self._centralize_grad(g)
            if self.use_clip:
                g = self._clip_grad(g)

            lr = self._get_lr(param_desc)

            # Initialize state if needed
            if p not in self.state:
                self.state[p] = {
                    "m": xp.zeros_like(p.data, dtype=DTYPE),   # first moment
                    "v": xp.zeros_like(p.data, dtype=DTYPE),   # second moment
                    "slow": p.data.copy(),                     # lookahead
                    "ema": p.data.copy() if self.use_ema else None,  # EMA
                }

            state = self.state[p]

            # Adam moments
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * g
            m_hat = state["m"] / (1 - xp.power(self.beta1, self.t))

            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (g * g)
            v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))

            # RAdam variance rectification
            r_t = 1.0
            if self.use_radam:
                beta2_t = xp.power(self.beta2, self.t)
                rho_t = self.rho_inf - 2.0 * self.t * beta2_t / (1.0 - beta2_t)
                if rho_t > 4:
                    r_t = xp.sqrt(
                        ((rho_t - 4.0) * (rho_t - 2.0) * self.rho_inf)
                        / ((self.rho_inf - 4.0) * (self.rho_inf - 2.0) * rho_t)
                    )

            # Update rule (AdamW style)
            update = r_t * m_hat / (xp.sqrt(v_hat) + self.epsilon)

            # Norm Loss scaling
            if self.use_norm_loss:
                w_norm = xp.linalg.norm(p.data)
                if w_norm > 0:
                    update /= w_norm

            # Decoupled weight decay
            if self.weight_decay > 0:
                p.data -= lr * self.weight_decay * p.data

            # Apply update
            p.data -= lr * update

            # Lookahead
            if self.step_counter % self.k == 0:
                state["slow"] += self.alpha * (p.data - state["slow"])
                p.data = state["slow"].copy()

            # EMA
            if self.use_ema:
                state["ema"] = self.ema_decay * state["ema"] + (1 - self.ema_decay) * p.data

    def swap_weights_with_ema(self, params):
        """Swap weights with EMA values (useful for evaluation)."""
        if not self.use_ema:
            return
        for param_desc in params:
            if isinstance(param_desc, dict):
                p = param_desc["param"]
            else:
                p = param_desc
            if p not in self.state or self.state[p]["ema"] is None:
                continue
            p.data, self.state[p]["ema"] = self.state[p]["ema"], p.data