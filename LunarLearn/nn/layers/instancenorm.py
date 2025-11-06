import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE

class InstanceNorm(BaseLayer):
    def __init__(self, ndim, epsilon=1e-5):
        super().__init__(trainable=True)
        self.ndim = ndim
        self.epsilon = xp.array(epsilon, dtype=DTYPE)

    def initialize(self, input_shape):
        n_C= input_shape[1]
        shape = (1, n_C, *([1] * self.ndim))
        
        W = xp.ones(shape, dtype=DTYPE)
        b = xp.zeros(shape, dtype=DTYPE)

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)
        self.output_shape = input_shape

    def _axis(self):
        return tuple(range(2, 2 + self.ndim))
    
    def forward(self, Z: Tensor) -> Tensor:
        if self.W is None or self.b is None:
            self.initialize(Z.shape[1:])

        W = self.W.to_compute()
        b = self.b.to_compute()

        axis = self._axis()
        mean = ops.mean(Z, axis=axis, keepdims=True)
        var = ops.var(Z, axis=axis, keepdims=True)
        denom = ops.sqrt(var + self.epsilon)
        inv_denom = 1.0 / denom
        Z_norm = (Z - mean) * inv_denom
        Z_out = W * Z_norm + b

        return Z_out