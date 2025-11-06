import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.optim.optimizers import BaseOptimizer

xp = backend.xp
DTYPE = backend.DTYPE

class Shampoo(BaseOptimizer):
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
        for param, layer, custom_optim in self._iter_params(params):
            # If param has its own optimizer → delegate and skip
            if custom_optim is not None and custom_optim is not self:
                custom_optim.step([param])
                continue
            # Otherwise, use *this* optimizer to update it
            grad = param.grad
            data = param.data
            lr = self._get_lr(param, layer)
            shape = data.shape
            rank = len(shape)

            # Initialize state
            if param not in self.state:
                self.state[param] = {
                    "preconds": [xp.eye(dim, dtype=DTYPE) for dim in shape],
                    "invs": [xp.eye(dim, dtype=DTYPE) for dim in shape],
                }
                if self.graft == "adam":
                    self.state[param].update({
                        "m": xp.zeros_like(data, dtype=DTYPE),
                        "v": xp.zeros_like(data, dtype=DTYPE),
                    })

            state = self.state[param]

            # Update preconditioners
            for mode in range(rank):
                axes = list(range(rank))
                axes[0], axes[mode] = axes[mode], axes[0]
                grad_unf = xp.reshape(xp.moveaxis(grad, mode, 0), (shape[mode], -1))
                cov = grad_unf @ grad_unf.T
                state["preconds"][mode] = self.beta * state["preconds"][mode] + (1 - self.beta) * cov

                if self.t % self.update_freq == 0:
                    eigvals, eigvecs = xp.linalg.eigh(state["preconds"][mode] + self.epsilon * xp.eye(shape[mode]))
                    inv_root = eigvecs @ xp.diag(1.0 / xp.sqrt(eigvals)) @ eigvecs.T
                    state["invs"][mode] = inv_root

            # Apply preconditioning
            g_pre = grad
            for mode in range(rank):
                inv_root = state["invs"][mode]
                g_pre = xp.tensordot(inv_root, g_pre, axes=[[1], [mode]])
                g_pre = xp.moveaxis(g_pre, 0, mode)

            # Grafting
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

            # Apply update
            data -= lr * scaled_update
            self._apply_weight_decay(param, layer, lr)