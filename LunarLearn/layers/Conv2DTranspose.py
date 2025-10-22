import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class Conv2DTranspose(BaseLayer):
    """
    2D Transposed Convolutional layer (a.k.a. deconvolution) with lazy 
    initialization and autograd support.

    This layer performs a 2D transposed convolution (gradient of Conv2D 
    with respect to its input) over an input tensor of shape 
    (batch, channels, height, width). It supports lazy initialization 
    of weights and biases, integrates with the autograd system, and 
    provides optional activation functions. Commonly used for upsampling 
    in autoencoders and generative models.

    Args:
        filters (int): 
            Number of output channels (convolution filters).
        kernel_size (int): 
            Size of the square convolution kernel (f x f).
        strides (int, optional): 
            Stride of the transposed convolution. Default is 1.
        padding (int or str, optional): 
            Padding mode. Default is 0.
            - int: number of zero-padding pixels around height and width
            - 'same': automatic padding to preserve output dimensions
        activation (str, optional): 
            Activation function applied after convolution. Default is 'linear'.  
            Supported: ['linear', 'sigmoid', 'relu', 'leaky_relu', 'tanh', 
            'softmax', 'log_softmax', 'swish', 'mish', 'gelu', 'softplus', 
            'elu', 'selu']
        w_init (str, optional): 
            Weight initialization strategy. Default is 'auto'.  
            Supported: ['He', 'Xavier', 'LeCun', 'Orthogonal', 'auto']
        uniform (bool, optional): 
            If True, use uniform distribution in initialization. Default is False.
        gain (float, optional): 
            Scaling factor applied during weight initialization. Default is 1.

    Attributes:
        W (Tensor): 
            Convolution weight tensor of shape (C_in, filters, f, f).
        b (Tensor): 
            Bias tensor of shape (filters, 1).
        output_shape (tuple): 
            Output tensor shape (filters, H_out, W_out), determined after initialization.

    Methods:
        initialize(input_shape):
            Lazily initializes weights and biases given an input shape (C_in, H, W).
        forward(A_prev: Tensor) -> Tensor:
            Performs the transposed convolution followed by the specified 
            activation function.

            Returns:
                Tensor: Activated output tensor of shape 
                (batch, filters, H_out, W_out).
    """
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear', w_init='auto', uniform=False, gain=1, groups=1): #add default values

        # Validate number of filters
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError("filters must be a positive integer")

        # Validate kernel size
        if not isinstance(kernel_size, (int, tuple)):
            raise ValueError("kernel_size must be int or tuple")
        if isinstance(kernel_size, int) and kernel_size < 0:
            raise ValueError("kernel_size must be < 0")
        else:
            if kernel_size[0] or kernel_size[1] < 0:
                raise ValueError("kernel_size dimensions must be < 0")

        # Validate stride
        if not isinstance(strides, int) or strides <= 0:
            raise ValueError("strides must be > 0")

        # Validate padding
        if not (padding == 'same' or isinstance(padding, int)):
            raise ValueError("padding must be either 'same' or an integer")
        if isinstance(padding, int) and padding < 0:
            raise ValueError("padding must be >= 0")

        # Validate gain
        if not isinstance(gain, (int, float)) or gain <= 0:
            raise ValueError("gain must be a positive number")
        
        # validate uniform
        if not isinstance(uniform, bool):
            raise ValueError("uniform must be True or False (boolean)")

        # Validate activation
        allowed_activations = ['linear', 'sigmoid', 'ReLU', 'leaky_ReLU', 'tanh', 'softmax', 'auto']
        if activation not in allowed_activations:
            raise ValueError(f"Unsupported activation '{activation}'. "
                             f"Available: {allowed_activations}")
        
        # Validate w_init
        allowed_w_init = ['He', 'Xavier', 'LeCun', 'orthogonal', 'auto']
        if w_init not in allowed_w_init:
            raise ValueError(f"Unsupported weight initialization '{w_init}'. "
                             f"Available: {allowed_w_init}")
        
        super().__init__(trainable=True)
        self.filters = filters
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain
        self.groups = groups

    def initialize(self, input_shape):
        from LunarLearn.engine import initialize_weights

        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")
        
        n_C_prev, n_H_prev, n_W_prev = input_shape
        f_h, f_w = self.kernel_size

        W, b = initialize_weights((n_C_prev, self.filters, f_h, f_w),
                                            (self.filters, 1),
                                            self.w_init, self.activation,
                                            self.uniform, self.gain)

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)

        if self.padding == 'same':
            self.padding = f_h // 2
        self.n_H = int((n_H_prev - 1) * self.strides - 2 * self.padding + f_h)
        self.n_W = int((n_W_prev - 1) * self.strides - 2 * self.padding + f_w)
        self.n_C_prev = n_C_prev
        self.output_shape = (self.filters, self.n_H, self.n_W)

    def forward(self, A_prev: Tensor) -> Tensor:
        """
        Compute the forward pass of the 2D transposed convolutional layer.

        This method performs a 2D transposed convolution (upsampling),
        applies the specified activation function, and supports mixed precision.

        Args:
            A_prev (Tensor): Input tensor of shape (N, C_in, H_in, W_in).
            training (bool, optional): Indicates whether the layer is in
                training mode. Currently has no effect.

        Returns:
            Tensor: Output tensor of shape (N, filters, H_out, W_out) after
                transposed convolution and activation.
        """
        from LunarLearn.activations import get_activation

        if self.W is None or self.b is None:
            _, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
            self.initialize((n_C_prev, n_H_prev, n_W_prev))

        W = self.W.to_compute()
        b = self.b.to_compute()

        m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
        output_shape = (m, self.filters, self.n_H, self.n_W)

        if self.groups == 1:
            X_col = ops.im2col_transpose(A_prev, self.kernel_size, self.strides, output_shape)
            W_col = W.reshape(n_C_prev, -1).T
            out_col = ops.matmul(W_col, X_col)
            z = ops.col2im_transpose(out_col, A_prev.shape, self.kernel_size, self.strides, output_shape)
        else:
            X_col = ops.im2col_transpose_grouped(A_prev, self.kernel_size, self.strides, output_shape, self.groups)
            W_col = W.reshape(self.groups, self.group_in_channels, -1)
            W_col = W_col.transpose(0, 2, 1).reshape(-1, self.group_in_channels)  # concat groups for matmul
            out_col = ops.matmul(W_col, X_col)
            z = ops.col2im_transpose_grouped(out_col, A_prev.shape, self.kernel_size, self.strides, output_shape, self.groups)
        z += b

        activation = get_activation(self.activation)
        A = activation(z)

        return A