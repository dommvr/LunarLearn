import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn import activations
from LunarLearn.activations import get_activation

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class Activation(BaseLayer):
    """
    Generic activation layer with autograd and mixed precision support.

    This layer applies the specified activation function to its input tensor.
    The available activation functions are defined in `LunarLearn.activations.ACTIVATIONS`.

    Args:
        activation (str):
            Name of the activation function to apply. Must be one of the keys
            in `LunarLearn.activations.ACTIVATIONS`.

    Attributes:
        activation (str):
            The chosen activation function name.
        output_shape (tuple):
            Output tensor shape, identical to input shape.

    Methods:
        initialize(input_shape):
            Stores the output shape as identical to input_shape.
        forward(Z: Tensor) -> Tensor:
            Applies the selected activation function.
    """

    def __init__(self, activation: str = "linear"):
        super().__init__(trainable=False)

        # Validate activation
        allowed_activations = activations.ACTIVATIONS
        if activation not in allowed_activations:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Available: {list(allowed_activations.keys())}"
            )

        self.activation = activation
        self._func = get_activation(activation)

    def initialize(self, input_shape):
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")
        self.output_shape = input_shape

    def forward(self, Z: Tensor) -> Tensor:
        """
        Compute the forward pass of the activation layer.

        Args:
            Z (Tensor): Input tensor of any shape (N, ...).
            training (bool, optional): Indicates training mode. 
                Does not affect behavior of activation functions.

        Returns:
            Tensor: Output tensor of the same shape as input with the activation applied.
        """
        A = self._func(Z)

        return A
