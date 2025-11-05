from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor

class Flatten(BaseLayer):
    """
    Flatten layer that reshapes input from (N, C, H, W) into (N, C*H*W).

    This layer is typically used to bridge convolutional layers with fully
    connected (Dense) layers. Reshaping is tracked by autograd so gradients
    flow correctly without a custom backward implementation.

    Attributes:
        nodes (int): 
            Total number of features per sample after flattening.
        output_shape (tuple): 
            Shape of the output tensor after flattening, excluding the batch dim.

    Methods:
        initialize(input_shape):
            Lazily initializes the layer based on input shape (C, H, W).
        forward(A_prev: Tensor) -> Tensor:
            Flattens the input tensor to 2D (batch, features) and returns it.
    """
    def __init__(self):
        
        super().__init__(trainable=False)

    def initialize(self, input_shape):
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")
        
        n_C_prev, n_H_prev, n_W_prev = input_shape
        self.nodes = n_C_prev * n_H_prev * n_W_prev
        self.output_shape = (self.nodes,)

    def forward(self, A_prev: Tensor) -> Tensor:
        """
        Perform the forward pass: flatten input to 2D.

        Args:
            A_prev (Tensor): Input tensor of shape (N, C, H, W).
            training (bool, optional): Unused, kept for API consistency.

        Returns:
            Tensor: Flattened tensor of shape (N, C*H*W).
        """
        return A_prev.reshape(A_prev.shape[0], -1)