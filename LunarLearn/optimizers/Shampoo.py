import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Shampoo(BaseOptimizer):
    """
    Shampoo optimizer with optional grafting (SGD/Adam).

    Shampoo preconditions gradients with matrix inverse roots of their
    second-order statistics along each tensor mode. It can be combined
    with grafting (e.g., Adam or SGD) to stabilize updates.

    Args:
        learning_rate (float, optional):
            Base learning rate. Default is 0.001.
        beta (float, optional):
            Exponential decay for covariance estimates. Default is 0.9.
        epsilon (float, optional):
            Numerical stability constant for inverses. Default is 1e-12.
        update_freq (int, optional):
            Frequency (in steps) to recompute inverse roots. Default is 10.
        graft (str, optional):
            Grafting strategy: 'adam' or 'sgd'. Default is 'adam'.
        beta2 (float, optional):
            Exponential decay for second moment (Adam graft). Default is 0.999.

    Attributes:
        state (dict):
            Per-parameter optimizer state including preconditioners, inverse
            roots, and optional Adam moments.
        t (int):
            Global timestep.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Tensors and dicts containing {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-12,
                 update_freq=10, graft="adam", beta2=0.999):
        super().__init__(learning_rate)
        self.beta = xp.array(beta, dtype=DTYPE)
        self.beta2 = xp.array(beta2, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.update_freq = update_freq
        self.graft = graft
        self.t = 0
        self.state = {}

    def step(self, params):
        self.t += 1

        for param_desc in params:
            # --- Extract param & layer ---
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
                shape = p.data.shape
                rank = len(shape)

                self.state[p] = {
                    "preconds": [xp.eye(dim, dtype=DTYPE) for dim in shape],
                    "invs": [xp.eye(dim, dtype=DTYPE) for dim in shape],
                }

                if self.graft == "adam":
                    self.state[p].update({
                        "m": xp.zeros_like(p.data, dtype=DTYPE),
                        "v": xp.zeros_like(p.data, dtype=DTYPE),
                    })

            state = self.state[p]
            shape = grad.shape
            rank = len(shape)

            # Update Shampoo preconditioners
            for mode in range(rank):
                axes = list(range(rank))
                axes[0], axes[mode] = axes[mode], axes[0]
                grad_unf = xp.reshape(xp.moveaxis(grad, mode, 0), (shape[mode], -1))

                cov = grad_unf @ grad_unf.T
                state["preconds"][mode] = (
                    self.beta * state["preconds"][mode] + (1 - self.beta) * cov
                )

                if self.t % self.update_freq == 0:
                    eigvals, eigvecs = xp.linalg.eigh(
                        state["preconds"][mode] + self.epsilon * xp.eye(shape[mode])
                    )
                    inv_root = eigvecs @ xp.diag(1.0 / xp.sqrt(eigvals)) @ eigvecs.T
                    state["invs"][mode] = inv_root

            # Preconditioned gradient
            g_pre = grad
            for mode in range(rank):
                inv_root = state["invs"][mode]
                g_pre = xp.tensordot(inv_root, g_pre, axes=[[1], [mode]])
                g_pre = xp.moveaxis(g_pre, 0, mode)

            # Grafting step
            if self.graft == "sgd":
                graft_step = grad
            elif self.graft == "adam":
                state["m"] = self.beta * state["m"] + (1 - self.beta) * grad
                state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad * grad)

                m_hat = state["m"] / (1 - xp.power(self.beta, self.t))
                v_hat = state["v"] / (1 - xp.power(self.beta2, self.t))
                graft_step = m_hat / (xp.sqrt(v_hat) + self.epsilon)
            else:
                raise ValueError(f"Unknown graft type: {self.graft}")

            # Norm scaling
            graft_norm = xp.linalg.norm(graft_step)
            shampoo_norm = xp.linalg.norm(g_pre) + self.epsilon
            scaled_update = g_pre * (graft_norm / shampoo_norm)

            # Parameter update
            p.data -= lr * scaled_update