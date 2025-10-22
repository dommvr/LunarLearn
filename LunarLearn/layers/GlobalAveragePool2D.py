import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class GlobalAveragePool2D(BaseLayer):
    """
    Global Average Pooling layer for 2D feature maps.

    This layer computes the mean value across the spatial dimensions (H, W)
    for each channel, reducing (N, C, H, W) input to (N, C). It is often used
    before fully connected layers to reduce parameters and spatial dimensions.

    Attributes:
        n_C (int):
            Number of channels in the input tensor.
        output_shape (tuple):
            Output tensor shape (C,) after pooling.

    Methods:
        initialize(input_shape):
            Sets up the layer based on input shape (C, H, W).
        forward(A_prev: Tensor) -> Tensor:
            Computes the mean over spatial dimensions and returns (N, C).
    """
    def __init__(self):
        super().__init__(trainable=False)

    def initialize(self, input_shape):
        # input_shape: (C, H, W)
        n_C, n_H, n_W = input_shape
        self.filters = n_C
        self.output_shape = (n_C,)

    def forward(self, A_prev: Tensor) -> Tensor:
        """
        Compute global average pooling over spatial dimensions.

        Args:
            A_prev (Tensor): Input tensor of shape (N, C, H, W).
            training (bool, optional): Indicates training mode. Currently unused
                but included for API consistency.

        Returns:
            Tensor: Tensor of shape (N, C) containing spatial averages.
        """
        
        # Compute mean over H and W
        A = ops.mean(A_prev, axis=(2, 3))

        return A
