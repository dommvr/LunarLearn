import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp


class Conv2D(BaseLayer):
    """
    2D Convolutional layer with lazy initialization and autograd support.

    This layer performs a 2D convolution over an input tensor with shape
    (batch, channels, height, width). It supports lazy initialization of
    weights and biases, integrates with the autograd system, and provides
    optional activation functions.

    Args:
        filters (int): 
            Number of output channels (convolution filters).
        kernel_size (int): 
            Size of the square convolution kernel (f x f).
        strides (int, optional): 
            Stride of the convolution. Default is 1.
        padding (int or str, optional): 
            Padding mode. Default is 0.
            - int: number of zero-padding pixels around height and width
            - 'same': automatic padding to preserve spatial dimensions
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
            Convolution weight tensor of shape (f, f, in_channels, filters).
        b (Tensor): 
            Bias tensor of shape (filters, 1).
        output_shape (tuple): 
            Output tensor shape (filters, H_out, W_out), determined after initialization.

    Methods:
        initialize(input_shape):
            Lazily initializes weights and biases given an input shape (C_in, H, W).
        forward(A_prev: Tensor) -> Tensor:
            Performs the convolution followed by the specified activation function.

            Returns:
                Tensor: Activated output tensor of shape 
                (batch, filters, H_out, W_out).
    """
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, dilation=1, bias=True):
        from LunarLearn.nn.activations import activations
        from LunarLearn.nn.initializations import initializations
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
        
        # Validate groups
        if not isinstance(groups, int) or groups < 1:
            raise ValueError("groups must be >= 1")

        # Validate activation
        allowed_activations = activations.ACTIVATIONS
        if activation not in allowed_activations:
            raise ValueError(f"Unsupported activation '{activation}'. "
                             f"Available: {list(allowed_activations.keys())}")
        
        # Validate w_init
        allowed_inits = initializations.ALL_INITIALIZATIONS
        if w_init not in allowed_inits:
            raise ValueError(f"Unsupported weight initialization '{w_init}'. "
                             f"Available: {list(allowed_inits.keys())}")

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
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
        self.bias = bias

    def initialize(self, input_shape):
        from LunarLearn.nn.initializations import initialize_weights

        # Validate input_shape 
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")

        n_C_prev, n_H_prev, n_W_prev = input_shape

        if n_C_prev % self.groups != 0:
            raise ValueError(f"Input channels ({n_C_prev}) must be divisible by groups ({self.groups})")
        if self.filters % self.groups != 0:
            raise ValueError(f"Output filters ({self.filters}) must be divisible by groups ({self.groups})")


        f_h, f_w = self.kernel_size
        d_h, d_w = self.dilation
        group_in_channels = n_C_prev // self.groups
        group_out_channels = self.filters // self.groups

        W, b = initialize_weights((self.filters, group_in_channels, f_h, f_w),
                                  (self.filters, 1),
                                  self.w_init, self.activation,
                                  self.uniform, self.gain)
        
        self.W = Parameter(W, requires_grad=True)
        if self.bias:
            self.b = Parameter(b, requires_grad=True)
        else:
            self.b = None

        self._apply_param_settings()

        eff_kh = d_h * (f_h - 1) + 1
        eff_kw = d_w * (f_w - 1) + 1

        if self.padding == 'same':
            self.padding_h = ((self.strides-1) * n_H_prev + eff_kh - self.strides) // 2
            self.padding_w = ((self.strides-1) * n_W_prev + eff_kw - self.strides) // 2
        else:
            self.padding_h = self.padding_w = self.padding

        self.n_H = (n_H_prev + 2*self.padding_h - eff_kh) // self.strides + 1
        self.n_W = (n_W_prev + 2*self.padding_w - eff_kw) // self.strides + 1
        self.output_shape = (self.filters, self.n_H, self.n_W)

    def forward(self, A_prev: Tensor) -> Tensor:
        """
        Compute the forward pass of the 2D convolutional layer.

        This method performs a 2D convolution over the input tensor, applies
        the specified activation function, and supports mixed precision.

        Args:
            A_prev (Tensor): Input tensor of shape (N, C_in, H, W).
            training (bool, optional): Indicates whether the layer is in
                training mode. Currently affects nothing as dropout/batchnorm
                is not applied here.

        Returns:
            Tensor: Output tensor of shape (N, filters, H_out, W_out) after
                convolution and activation.
        """
        from LunarLearn.nn.activations import get_activation

        if self.W is None:
            _, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
            self.initialize((n_C_prev, n_H_prev, n_W_prev))

        W = self.W.to_compute()
        b = self.b.to_compute()

        m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
        pad_h, pad_w = self.padding_h, self.padding_w
        d_h, d_w = self.dilation

        # Pad input
        A_prev_pad = ops.pad(A_prev,
                            ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)),
                            mode='constant')

        # Grouped or standard im2col
        if self.groups > 1:
            X_col = ops.im2col_grouped(A_prev_pad, self.kernel_size, self.strides, self.groups, self.dilation)
        else:
            X_col = ops.im2col(A_prev_pad, self.kernel_size, self.strides, self.dilation)

        # Reshape W for matmul
        W_col = W.reshape(self.filters, -1)
        # Matrix multiplication
        Z_col = ops.matmul(W_col, X_col)

        if self.b is not None:
            b = self.b.to_compute()
            Z_col += b

        # Reshape to output
        H_out = self.n_H
        W_out = self.n_W
        Z = Z_col.reshape(self.filters, H_out, W_out, m).transpose(3,0,1,2)

        # Activation
        activation = get_activation(self.activation)
        A = activation(Z)

        return A