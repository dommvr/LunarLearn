import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class ReLU(BaseLayer):
    """
    ReLU (Rectified Linear Unit) activation layer with autograd support.

    Applies the element-wise function ReLU(x) = max(0, x) to the input tensor.
    Compatible with mixed precision and integrates seamlessly with autograd.

    Attributes:
        output_shape (tuple):
            Shape of the output tensor, identical to the input shape.

    Methods:
        initialize(input_shape):
            Stores the output shape as identical to input_shape.
        forward(Z: Tensor) -> Tensor:
            Applies the ReLU activation function element-wise.
    """
    def __init__(self):
        
        super().__init__(trainable=False)

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
        """
        Compute the forward pass of the ReLU activation layer.

        Args:
            Z (Tensor): Input tensor of any shape (N, ...).
            training (bool, optional): Indicates training mode. Does not affect behavior.

        Returns:
            Tensor: Output tensor of the same shape as input, with ReLU applied.
        """
        A = ops.maximum(Z, 0)

        return A