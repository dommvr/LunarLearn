import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Sophia(BaseOptimizer):
    """
    Sophia optimizer with Hessian-based preconditioning.

    Sophia uses a low-rank Hessian approximation (either per-weight or
    per-layer trace) to scale updates. It clips the Hessian estimate with
    a threshold (rho) for stability.

    Args:
        learning_rate (float, optional):
            Base learning rate. Default is 0.001.
        beta1 (float, optional):
            Momentum coefficient for first moment. Default is 0.965.
        beta2 (float, optional):
            Momentum coefficient for Hessian estimation. Default is 0.99.
        rho (float, optional):
            Maximum Hessian value (clipping threshold). Default is 0.04.
        epsilon (float, optional):
            Numerical stability term. Default is 1e-8.
        mode (str, optional):
            Approximation mode:
            - "G": diagonal (per-weight) Hessian estimate.
            - "H": scalar (per-layer trace) Hessian estimate.
            Default is "G".

    Attributes:
        state (dict):
            Per-parameter state storing momentum and Hessian estimates.
        t (int):
            Global timestep.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Tensors and dicts containing {"param": Tensor, "layer": Layer}.
    """
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

        for param_desc in params:
            # --- Extract param ---
            if isinstance(param_desc, dict):
                p = param_desc["param"]
            else:
                p = param_desc

            if not isinstance(p, Tensor) or not p.requires_grad:
                continue
            if p.grad is None:
                continue

            grad = p.grad
            lr = self._get_lr(param_desc)

            # Initialize state
            if p not in self.state:
                state = {"m": xp.zeros_like(p.data, dtype=DTYPE)}
                if self.mode == "G":
                    state.update({"h": xp.zeros_like(p.data, dtype=DTYPE)})
                elif self.mode == "H":
                    state.update({"h": xp.array(0.0, dtype=DTYPE)})
                self.state[p] = state

            state = self.state[p]

            # Momentum update
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad

            if self.mode == "G":
                # Per-weight Hessian approx
                state["h"] = self.beta2 * state["h"] + (1 - self.beta2) * (grad * grad)
                h_clipped = xp.minimum(state["h"], self.rho)
                p.data -= lr * state["m"] / (h_clipped + self.epsilon)

            elif self.mode == "H":
                # Per-layer scalar Hessian trace
                h_trace = xp.mean(grad * grad)
                state["h"] = self.beta2 * state["h"] + (1 - self.beta2) * h_trace
                h_clipped = xp.minimum(state["h"], self.rho)
                p.data -= lr * state["m"] / (h_clipped + self.epsilon)