import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor
from LunarLearn.nn.activations import leaky_relu

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class LeakyReLU(BaseLayer):
    def __init__(self, alpha=0.01):
        super().__init__(trainable=False)
        self.alpha = alpha

    def initialize(self, input_shape):
        # Validate input_shape
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")
    
        n_C_prev, n_H_prev, n_W_prev = input_shape

        self.n_C = n_C_prev
        self.n_H = n_H_prev
        self.n_W = n_W_prev
        self.output_shape = (self.n_C, self.n_H, self.n_W)

    def forward(self, Z: Tensor) -> Tensor:
        A = leaky_relu(Z, alpha=self.alpha)
        return A