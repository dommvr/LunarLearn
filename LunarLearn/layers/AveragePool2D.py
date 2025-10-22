import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class AveragePool2D(BaseLayer):
    """
    2D Average Pooling layer with autograd support.

    This layer performs downsampling by computing the average over 
    non-overlapping or strided patches of the input tensor. Supports
    mixed precision and integrates with the autograd system.

    Args:
        pool_size (int): Size of the square pooling window (f x f).
        strides (int): Stride of the pooling operation.
        padding (int or str, optional): Padding mode. Default is 0.
            - int: number of zero-padding pixels around height and width
            - 'same': automatic padding to preserve spatial dimensions

    Attributes:
        pool_size (int): Pooling window size.
        strides (int): Pooling stride.
        padding (int): Padding applied to height and width.
        n_C (int): Number of channels in input.
        n_H (int): Height of output after pooling.
        n_W (int): Width of output after pooling.
        output_shape (tuple): Output tensor shape (C, H_out, W_out).

    Methods:
        initialize(input_shape):
            Lazily computes output dimensions given an input shape (C, H, W).
        forward(A_prev: Tensor) -> Tensor:
            Performs average pooling on the input tensor.
    """
    def __init__(self, pool_size, strides, padding=0):

        # Validate pool_size
        if not isinstance(pool_size, int):
            raise ValueError("kernel_size must be an integer")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        
        # Validate strides
        if not isinstance(strides, int):
            raise ValueError("strides must be an integer")
        if strides <= 0:
            raise ValueError("strides must be > 0")
        
        # Validate padding
        if not (padding == 'same' or isinstance(padding, int)):
            raise ValueError("padding must be either 'same' or an integer")
        if isinstance(padding, int) and padding < 0:
            raise ValueError("padding must be >= 0")

        super().__init__(trainable=False)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def initialize(self, input_shape):

        # Validate input_shape 
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")

        n_C_prev, n_H_prev, n_W_prev = input_shape
        f = self.pool_size

        if self.padding == 'same':
            self.padding = ((self.strides-1) * n_H_prev + f - self.strides) // 2

        self.n_C = n_C_prev
        self.n_H = int((n_H_prev + 2 * self.padding - f) / self.strides) + 1
        self.n_W = int((n_W_prev + 2 * self.padding - f) / self.strides) + 1
        self.output_shape = (self.n_C, self.n_H, self.n_W)

    def forward(self, A_prev: Tensor) -> Tensor:
        """
        Compute the forward pass of average pooling.

        Args:
            A_prev (Tensor): Input tensor of shape (N, C, H, W).
            training (bool, optional): Indicates whether in training mode. 
                                    Currently not used as pooling has no trainable parameters.

        Returns:
            Tensor: Output tensor after average pooling of shape (N, C, H_out, W_out).
        """
        if self.output_shape is None:
            _, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
            self.initialize((n_C_prev, n_H_prev, n_W_prev))

        # A_prev: (N, C, H, W)
        self.m, self.n_C, H_prev, W_prev = A_prev.shape
        f = self.pool_size
        pad = self.padding

        # Pad input
        A_prev_pad = ops.pad(A_prev,
                            ((0,0),(0,0),(pad,pad),(pad,pad)),
                            mode='constant')

        cols = ops.im2col(A_prev_pad, f, self.strides)
        cols_reshaped = cols.reshape(f * f, self.n_C, -1)

        out = ops.mean(cols_reshaped, axis=0)  # shape: (C, N*out_h*out_w)
        out = out.reshape(self.n_C, self.n_H, self.n_W, self.m).transpose(3, 0, 1, 2)  # (N, C, H, W)

        A = out

        return A