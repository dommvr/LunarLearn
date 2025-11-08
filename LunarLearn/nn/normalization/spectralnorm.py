import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Stateful
from LunarLearn.core import Tensor, Parameter

xp = backend.xp
DTYPE = backend.DTYPE

class SpectralNorm(Stateful):
    def __init__(self, param: Parameter, dim=0, n_iter=1, epsilon=1e-12):
        """
        Spectral Normalization constraint.
        Args:
            param: Parameter to normalize.
            dim: Dimension to treat as output (for convs: usually 0).
            n_iter: Number of power iterations per forward.
            epsilon: Numerical stability.
        """
        self.param = param
        self.dim = dim
        self.n_iter = n_iter
        self.epsilon = epsilon

        W = param.master
        self.W_shape = W.shape

        # Flatten to 2D (out_features, -1)
        self._reshape_for_matmul = lambda w: w.reshape(w.shape[0], -1)

        # Initialize power iteration vector u (not trainable)
        u = xp.random.randn(W.shape[0], 1)
        u /= xp.linalg.norm(u) + epsilon
        self.u = Tensor(u, requires_grad=False, dtype=param.master.dtype)

        # Tell the parameter it’s normalized by this
        param.normalization = self

    def state_dict(self):
        return {"u_data": self.u.data}
    
    def load_state_dict(self, state):
        if "u_data" in state:
            self.u.data[...] = xp.array(state["u_data"])

    def __call__(self, W: Tensor) -> Tensor:
        """Apply spectral normalization each forward."""
        W_mat = self._reshape_for_matmul(W.data)
        u = self.u

        for _ in range(self.n_iter):
            v = W_mat.T @ u
            v /= xp.linalg.norm(v) + self.epsilon
            u = W_mat @ v
            u /= xp.linalg.norm(u) + self.epsilon

        # Update the stored u vector (not tracked by autograd)
        self.u.data = u

        sigma = float((u.T @ W_mat @ v))
        W_bar = W_mat / (sigma + self.epsilon)

        out = Tensor(W_bar.reshape(self.W_shape), requires_grad=W.requires_grad)
        out.skip_grad = True  # <--- important, to bypass autograd chain
        return out