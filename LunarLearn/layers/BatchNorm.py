import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class BatchNorm(BaseLayer):
    """
    2D Batch Normalization layer with momentum and mixed precision support.

    This layer normalizes activations across the batch and spatial dimensions 
    for each channel, optionally maintaining running mean and variance for inference. 
    Supports mixed precision training and integrates with autograd.

    Parameters
    ----------
    momentum : float, optional
        Momentum factor for running mean and variance. Default is 0.9.
        Must be in the range (0, 1).
    epsilon : float, optional
        Small constant added to variance for numerical stability. Default is 1e-3.

    Attributes
    ----------
    W : Tensor
        Learnable scale parameter with shape (1, channels, 1, 1).
    b : Tensor
        Learnable shift parameter with shape (1, channels, 1, 1).
    running_mean : Tensor
        Running mean of shape (1, channels, 1, 1), updated during training.
    running_var : Tensor
        Running variance of shape (1, channels, 1, 1), updated during training.
    output_shape : tuple
        Output tensor shape (channels, height, width), determined after initialization.

    Methods
    -------
    initialize(input_shape)
        Initializes scale, shift, running mean, and running variance parameters.
    forward(Z: Tensor) -> Tensor
        Normalizes input activations, applies scale and shift, and returns the result.
    """
    def __init__(self, ndim, momentum=0.9, epsilon=1e-3):

        # Validate momentum
        if not isinstance(momentum, (float, int)):
            raise ValueError("momentum must be a float")
        if not (0 < momentum < 1):
            raise ValueError("momentum must be in the range (0, 1)")

        # Validate epsilon
        if not isinstance(epsilon, (float, int)):
            raise ValueError("epsilon must be a float")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        
        super().__init__(trainable=True)

        self.ndim = ndim

        self.momentum = xp.array(momentum, dtype=DTYPE)
        self.epsilon = xp.array(epsilon, dtype=DTYPE)

        self.custom_hook_metrics = ["running_mean", "running_var"]

    def initialize(self, input_shape):

        # Validate input_shape 
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")
        
        n_C_prev = input_shape[0]
        shape = (1, n_C_prev, *([1] * self.ndim))

        W = xp.ones(shape, dtype=DTYPE)
        b = xp.zeros(shape, dtype=DTYPE)

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)

        self.running_mean = xp.zeros(shape, dtype=DTYPE)
        self.running_var = xp.ones(shape, dtype=DTYPE)

        self.output_shape = input_shape[1:]

    def _axis(self):
        return (0,) + tuple(range(2, 2 + self.ndim))

    def forward(self, Z: Tensor) -> Tensor:
        """
        Perform batch normalization on a 4D input tensor.

        Parameters
        ----------
        Z : Tensor
            Input tensor of shape (batch_size, channels, height, width).
        training : bool, optional
            If True, use batch statistics and update running mean/variance.
            If False, use stored running mean and variance. Default is True.

        Returns
        -------
        Tensor
            Normalized, scaled, and shifted output tensor of same shape as input.
            Supports mixed precision training (casts to float32 if needed for stability).

        Notes
        -----
        - Autograd compatible: all operations use tensor ops, so gradients are tracked automatically.
        - Mixed precision: if MIXED_PRECISION is True, input is temporarily cast to float32
        for numerical stability during normalization.
        """
        W = self.W.to_compute()
        b = self.b.to_compute()

        if self.training:
            
            axis = self._axis()
            mean = ops.mean(Z, axis=axis, keepdims=True)
            var = ops.var(Z, axis=axis, keepdims=True)

            self.running_mean *= self.momentum
            self.running_mean += (1 - self.momentum) * mean.data
            self.running_var *= self.momentum
            self.running_var += (1 - self.momentum) * var.data

            Z_centered = Z - mean
            std_inv = 1.0 / ops.sqrt(var + self.epsilon)

            Z_norm = Z_centered * std_inv

            A = W * Z_norm + b

        else:
            
            denom = ops.sqrt(self.running_var + self.epsilon)
            inv_denom = 1.0 / denom
            Z_norm = (Z - self.running_mean) * inv_denom
            A = W * Z_norm + b

        return A