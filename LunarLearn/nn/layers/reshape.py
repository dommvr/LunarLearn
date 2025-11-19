from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor

class Reshape(BaseLayer):
    def __init__(self, *shape):
        super().__init__(trainable=False)
        self.shape = shape

    def initialize(self, input_shape):
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")
        
        n_C_prev, n_H_prev, n_W_prev = input_shape
        self.nodes = n_C_prev * n_H_prev * n_W_prev
        self.output_shape = (self.nodes,)

    def forward(self, A_prev: Tensor) -> Tensor:
        return A_prev.reshape(*self.shape)