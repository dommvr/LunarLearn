import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE

class RMSNorm(BaseLayer):
    def __init__(self, normalize_shape, epsilon=1e-5):
        super().__init__(trainable=True)
        self.normalize_shape = normalize_shape # e.g. (C,) or (C, H, W)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)

    def initialize(self, input_shape):
        W = xp.ones(self.normalize_shape, dtype=DTYPE)

        self.W = Parameter(W, requires_grad=True)
        self.output_shape = input_shape

    def forward(self, Z: Tensor) -> Tensor:
        if self.W is None or self.b is None:
            self.initialize(Z.shape[1:])

        W = self.W.to_compute()

        axes = tuple(range(1, Z.ndim))
        rms = ops.sqrt(ops.mean(Z * Z, axis=axes, keepdims=True))
        inv_denom = 1.0 / rms
        Z_norm = Z * inv_denom
        Z_out = W * Z_norm

        return Z_out
