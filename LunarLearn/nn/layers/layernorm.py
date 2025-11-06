import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE
    
class LayerNorm(BaseLayer):
    def __init__(self, epsilon=1e-5, axis=None):
        super().__init__(trainable=True)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)
        self.axis = axis

    def initialize(self, input_shape):
        if self.axis is None:
            self.axis = tuple(range(1, len(input_shape)))
        param_shape = tuple(input_shape[i] for i in self.axis)
        W = xp.ones(param_shape, dtype=DTYPE)
        b = xp.zeros(param_shape, dtype=DTYPE)

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)
        self.output_shape = input_shape

    def forward(self, Z: Tensor) -> Tensor:
        if self.W is None or self.b is None:
            self.initialize(Z.shape)

        W = self.W.to_compute()
        b = self.b.to_compute()

        mean = ops.mean(Z, axis=self.axis, keepdims=True)
        var = ops.var(Z, axis=self.axis, keepdims=True)
        denom = ops.sqrt(var + self.epsilon)
        inv_denom = 1.0 / denom
        Z_norm = (Z - mean) * inv_denom
        Z_out = W * Z_norm + b

        return Z_out