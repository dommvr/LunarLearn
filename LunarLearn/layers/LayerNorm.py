import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE

class LayerNorm(BaseLayer):
    def __init__(self, normalize_shape, epsilon=1e-5):
        super().__init__(trainable=True)
        self.normalize_shape = normalize_shape
        self.epsilon = xp.array(epsilon, dtype=DTYPE)

    def initialize(self, input_shape):
        W = xp.ones(self.normalize_shape, dtype=DTYPE)
        b = xp.zeros(self.normalize_shape, dtype=DTYPE)

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)
        self.output_shape = input_shape

    def forward(self, Z: Tensor) -> Tensor:
        if self.W is None or self.b is None:
            self.initialize(Z.shape[1:])

        W = self.W.to_compute()
        b = self.b.to_compute()

        axes = tuple(range(1, len(Z.shape)))
        mean = ops.mean(Z, axis=axes, keepdims=True)
        var = ops.var(Z, axis=axes, keepdims=True)
        denom = ops.sqrt(var + self.epsilon)
        inv_denom = 1.0 / denom
        Z_norm = (Z - mean) * inv_denom
        Z_out = W * Z_norm + b

        return Z_out