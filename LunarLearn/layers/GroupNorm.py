import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE

class GroupNorm(BaseLayer):
    def __init__(self, num_groups, epsilon=1e-5):
        super().__init__(trainable=True)
        self.num_groups = num_groups
        self.epsilon = xp.array(epsilon, dtype=DTYPE)

    def initialize(self, input_shape):
        n_C = input_shape[1]
        if n_C % self.num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        
        shape = (1, n_C, *([1] * (len(input_shape) - 1)))
        
        W = xp.ones(shape, dtype=DTYPE)
        b = xp.zeros(shape, dtype=DTYPE)

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)
        self.output_shape = input_shape

    def forward(self, Z: Tensor) -> Tensor:
        if self.W is None or self.b is None:
            self.initialize(Z.shape[1:])

        W = self.W.to_compute()
        b = self.b.to_compute()

        N, n_C = Z.shape[:2]
        G = self.num_groups
        Z_reshaped = Z.reshape(N, G, n_C // G, *Z.shape[2:])
        axes = tuple(range(2, Z_reshaped.ndim))
        mean = ops.mean(Z_reshaped, axis=axes, keepdims=True)
        var = ops.var(Z_reshaped, axis=axes, keepdims=True)
        denom = ops.sqrt(var + self.epsilon)
        inv_denom = 1.0 / denom
        Z_norm_reshaped = (Z_reshaped - mean) * inv_denom
        Z_norm = Z_norm_reshaped.reshape(*Z.shape)
        Z_out = W * Z_norm + b

        return Z_out