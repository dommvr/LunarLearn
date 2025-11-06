import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class MaxPool2D(BaseLayer):
    """
    2D Max Pooling layer with lazy initialization and autograd support.

    This layer performs a 2D max pooling operation over an input tensor with shape
    (batch, channels, height, width). It supports lazy initialization of the output
    shape and uses only tensor operations, making it fully compatible with autograd.

    Args:
        pool_size (int): 
            Size of the square pooling window (f x f).
        strides (int): 
            Stride of the pooling operation.
        padding (int or str, optional): 
            Padding mode. Default is 0.
            - int: number of zero-padding pixels around height and width
            - 'same': automatic padding to preserve spatial dimensions

    Attributes:
        output_shape (tuple): 
            Output tensor shape (channels, H_out, W_out), determined after initialization.

    Methods:
        initialize(input_shape):
            Lazily computes the output shape given an input shape (C_in, H, W).
        forward(A_prev: Tensor) -> Tensor:
            Performs max pooling over the input tensor.

            Returns:
                Tensor: Output tensor of shape (batch, channels, H_out, W_out) 
                after max pooling.
    """
    def __init__(self, pool_size: int, strides: int, padding=0):
        if not isinstance(pool_size, int) or pool_size <= 0:
            raise ValueError("pool_size must be a positive integer")
        if not isinstance(strides, int) or strides <= 0:
            raise ValueError("strides must be a positive integer")
        if not (padding == 'same' or isinstance(padding, int)):
            raise ValueError("padding must be either 'same' or an integer")
        if isinstance(padding, int) and padding < 0:
            raise ValueError("padding must be >= 0")

        super().__init__(trainable=False)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.output_shape = None

    def initialize(self, input_shape):
        n_C_prev, n_H_prev, n_W_prev = input_shape
        f = self.pool_size

        if self.padding == 'same':
            self.padding = ((self.strides - 1) * n_H_prev + f - self.strides) // 2

        self.n_C = n_C_prev
        self.n_H = (n_H_prev + 2 * self.padding - f) // self.strides + 1
        self.n_W = (n_W_prev + 2 * self.padding - f) // self.strides + 1
        self.output_shape = (self.n_C, self.n_H, self.n_W)

    def forward(self, A_prev: Tensor) -> Tensor:
        """
        Compute the forward pass of the 2D max pooling layer.

        This method performs max pooling over the input tensor and is fully compatible
        with autograd. During backpropagation, gradients are propagated through
        the locations of the maximum values.

        Args:
            A_prev (Tensor): Input tensor of shape (N, C_in, H, W).
            training (bool, optional): Indicates whether the layer is in training mode.
                This parameter does not change the behavior as pooling has no training-specific behavior.

        Returns:
            Tensor: Output tensor of shape (N, C_in, H_out, W_out) after max pooling.
        """

        m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
        f = self.pool_size
        s = self.strides
        pad = self.padding

        # Pad input
        A_prev_pad = ops.pad(A_prev,
                            ((0,0),(0,0),(pad,pad),(pad,pad)),
                            mode='constant')

        # im2col: shape (f*f*C, H_out*W_out*N)
        cols = ops.im2col(A_prev_pad, f, s)
        cols_reshaped = cols.reshape(f*f, n_C_prev, -1)  # (f*f, C, N*H_out*W_out)

        # Max pooling
        out = ops.max(cols_reshaped, axis=0)
        A = out.reshape(n_C_prev, self.n_H, self.n_W, m).transpose(3,0,1,2)

        return A