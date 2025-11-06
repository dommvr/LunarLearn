import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


class GradientProcessor:
    """
    Advanced gradient preprocessing pipeline:
    - Centralization
    - Clipping (global, per-layer, by value)
    - Gradient accumulation & scaling
    - NaN/Inf detection and handling
    - Gradient noise regularization
    - EMA tracking of gradient norms

    Tracks internal state to decide when to apply optimizer step.
    """
    def __init__(self, options: dict):
        self.options = options
        self.accum_step = 0
        self.nan_inf_policy = options.get("nan_inf_policy", "skip")  # skip | zero | warn
        self.noise_std = options.get("grad_noise_std", 0.0)
        self.noise_decay = options.get("grad_noise_decay", 0.999)
        self.ema_decay = options.get("grad_ema_decay", 0.9)
        self.grad_norm_ema = None
        self.grad_norm_var = 0.0  # for adaptive diagnostics

    # ---------------------------------------------------------------------
    # --- State and accumulation tracking ---
    # ---------------------------------------------------------------------
    def step_ready(self) -> bool:
        """Whether accumulated grads are ready for optimizer step."""
        acc_steps = self.options.get("accumulation_steps", 1)
        return (self.accum_step + 1) % acc_steps == 0

    def reset(self):
        """Reset accumulation counter."""
        self.accum_step = 0

    # ---------------------------------------------------------------------
    # --- Gradient stability checks ---
    # ---------------------------------------------------------------------
    def _detect_invalid(self, model):
        """Detect NaN/Inf gradients."""
        for param_desc in model.parameters(with_layer=True):
            p = param_desc["param"]
            if p.grad is not None:
                if xp.any(xp.isnan(p.grad)) or xp.any(xp.isinf(p.grad)):
                    return True
        return False

    def _handle_invalid(self, model):
        """Handle invalid gradients according to policy."""
        policy = self.nan_inf_policy
        if policy == "zero":
            for param_desc in model.parameters(with_layer=True):
                p = param_desc["param"]
                if p.grad is not None:
                    mask = xp.isnan(p.grad) | xp.isinf(p.grad)
                    if xp.any(mask):
                        p.grad[mask] = 0.0
            print("Invalid gradients zeroed.")
        elif policy == "skip":
            print("NaN/Inf gradients detected — skipping optimizer step.")
        elif policy == "warn":
            print("Warning: NaN/Inf gradients detected.")

    # ---------------------------------------------------------------------
    # --- Gradient norm tracking (EMA) ---
    # ---------------------------------------------------------------------
    def _update_grad_norm_stats(self, model):
        """Track gradient norm with EMA for diagnostics."""
        total_norm = 0.0
        count = 0

        for param_desc in model.parameters(with_layer=True):
            p = param_desc["param"]
            if p.grad is not None:
                norm = float(xp.sqrt(xp.sum(p.grad.astype(DTYPE) ** 2)))
                total_norm += norm
                count += 1

        if count == 0:
            return

        avg_norm = total_norm / count

        if self.grad_norm_ema is None:
            self.grad_norm_ema = avg_norm
            self.grad_norm_var = 0.0
        else:
            diff = avg_norm - self.grad_norm_ema
            self.grad_norm_ema = self.ema_decay * self.grad_norm_ema + (1 - self.ema_decay) * avg_norm
            self.grad_norm_var = self.ema_decay * self.grad_norm_var + (1 - self.ema_decay) * (diff ** 2)

        # Optional live diagnostics
        if self.accum_step % 50 == 0:
            std = math.sqrt(self.grad_norm_var)
            print(f"[GradStats] EMA norm: {self.grad_norm_ema:.6f} ± {std:.6f}")

    # ---------------------------------------------------------------------
    # --- Gradient noise injection ---
    # ---------------------------------------------------------------------
    def _add_gradient_noise(self, model):
        """
        Inject Gaussian noise into gradients for better exploration.
        Noise std decays over time: std_t = base_std * (decay ^ step)
        """
        if self.noise_std <= 0:
            return

        step_factor = self.noise_decay ** self.accum_step
        noise_level = self.noise_std * step_factor

        for param_desc in model.parameters(with_layer=True):
            p = param_desc["param"]
            if p.grad is not None:
                noise = xp.random.normal(0, noise_level, size=p.grad.shape).astype(DTYPE)
                p.grad += noise

    # ---------------------------------------------------------------------
    # --- Core processing logic ---
    # ---------------------------------------------------------------------
    def process(self, model):
        """
        Preprocess gradients for current step.
        Returns:
            bool: whether gradients are valid and ready for optimizer step.
        """
        self.accum_step += 1
        opts = self.options
        params = model.parameters(with_layer=True)

        # --- 1. Gradient centralization ---
        if opts.get("centralize", False):
            for param_desc in params:
                p = param_desc["param"]
                if p.grad is not None and p.grad.ndim > 1:
                    axes = tuple(range(1, p.grad.ndim))
                    p.grad -= xp.mean(p.grad, axis=axes, keepdims=True)

        # --- 2. Clip by global norm ---
        clip_norm = opts.get("clip_norm")
        if clip_norm is not None:
            total_norm = xp.sqrt(sum(
                float(xp.sum(p.grad.astype(DTYPE) ** 2))
                for param_desc in params
                if (p := param_desc["param"]).grad is not None
            ))
            clip_coef = clip_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for param_desc in params:
                    p = param_desc["param"]
                    if p.grad is not None:
                        p.grad *= clip_coef

        # --- 3. Clip by per-layer norm ---
        clip_layer = opts.get("clip_norm_per_layer")
        if clip_layer is not None:
            for param_desc in params:
                p = param_desc["param"]
                if p.grad is None:
                    continue
                norm = xp.linalg.norm(p.grad)
                if norm > clip_layer:
                    p.grad *= clip_layer / (norm + 1e-6)

        # --- 4. Clip by absolute value ---
        clip_val = opts.get("clip_value")
        if clip_val is not None:
            for param_desc in params:
                p = param_desc["param"]
                if p.grad is not None:
                    p.grad = xp.clip(p.grad, -clip_val, clip_val)

        # --- 5. Add gradient noise (optional) ---
        self._add_gradient_noise(model)

        # --- 6. Detect NaN/Inf gradients ---
        if self._detect_invalid(model):
            self._handle_invalid(model)
            if self.nan_inf_policy == "skip":
                return False

        # --- 7. Gradient accumulation scaling ---
        acc_steps = opts.get("accumulation_steps", 1)
        if acc_steps > 1 and self.step_ready():
            for param_desc in params:
                p = param_desc["param"]
                if p.grad is not None:
                    p.grad /= acc_steps
            self.reset()  # reset after scaling

        # --- 8. Track gradient statistics ---
        self._update_grad_norm_stats(model)

        return True