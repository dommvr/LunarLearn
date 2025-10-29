import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class Dense(BaseLayer):
    """
    Fully connected (dense) layer with lazy initialization and autograd support.

    This layer applies a linear transformation followed by an optional activation
    function. It supports mixed precision training, lazy initialization of weights,
    and integrates with the autograd system. Dropout regularization is applied 
    after activation during training if keep_prob < 1.

    Parameters
    ----------
    nodes : int
        Number of output units (neurons).
    activation : str, default='linear'
        Activation function to apply after the linear transformation.
        Supported: ['linear', 'sigmoid', 'relu', 'leaky_relu', 'tanh', 
        'softmax', 'log_softmax', 'swish', 'mish', 'gelu', 'softplus', 
        'elu', 'selu']
    w_init : str, default='auto'
        Weight initialization strategy.
        Supported: ['He', 'Xavier', 'LeCun', 'Orthogonal', 'auto']
    uniform : bool, default=False
        If True, use uniform distribution in initialization (when supported).
    gain : float, default=1
        Scaling factor applied during weight initialization.
    keep_prob : float, default=1
        Probability of keeping a unit active during dropout (in range (0, 1]).

    Attributes
    ----------
    W : Tensor
        Weight matrix of shape (n_in, nodes), initialized lazily.
    b : Tensor
        Bias vector of shape (1, nodes).
    output_shape : tuple
        Output tensor shape (nodes,), determined after initialization.

    Methods
    -------
    initialize(input_shape)
        Lazily initializes weights and biases given an input shape (n_in,).
    forward(A_prev: Tensor) -> Tensor
        Computes the linear transformation, applies activation,
        and applies dropout during training if keep_prob < 1.
    """
    def __init__(self, nodes, activation='linear',
                 w_init='auto', uniform=False, gain=1, keep_prob=1, transpose_weight=False):
        from LunarLearn.activations import activations
        from LunarLearn.initializations import initializations

        # Validate nodes
        if not isinstance(nodes, int) or nodes <= 0:
            raise ValueError("nodes must be a positive integer")

        # Validate gain
        if not isinstance(gain, (int, float)) or gain <= 0:
            raise ValueError("gain must be a positive number")

        # Validate keep_prob
        if not isinstance(keep_prob, (int, float)):
            raise ValueError("keep_prob must be a float or int")
        if not (0 < keep_prob <= 1):
            raise ValueError("keep_prob must be in the range (0, 1]")

        # Validate uniform
        if not isinstance(uniform, bool):
            raise ValueError("uniform must be True or False (boolean)")

        # Validate activation
        allowed_activations = activations.ACTIVATIONS
        if activation not in allowed_activations:
            raise ValueError(f"Unsupported activation '{activation}'. "
                             f"Available: {list(allowed_activations.keys())}")

        # Validate weight initialization
        allowed_inits = initializations.ALL_INITIALIZATIONS
        if w_init not in allowed_inits:
            raise ValueError(f"Unsupported weight initialization '{w_init}'. "
                             f"Available: {list(allowed_inits.keys())}")

        super().__init__(trainable=True)
        self.nodes = nodes
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain
        self.keep_prob = keep_prob
        self.transpose_weight = transpose_weight

    def initialize(self, input_shape):
        from LunarLearn.initializations import initialize_weights

        # Validate input_shape 
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")

        n_in = input_shape[0]

        W, b = initialize_weights(
            (self.nodes, n_in),
            (self.nodes, 1),
            self.w_init,
            self.activation,
            self.uniform,
            self.gain
        )

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True)

        self._apply_param_settings()

        self.output_shape = (self.nodes,)

    def forward(self, A_prev: Tensor) -> Tensor:
        """
        Compute the forward pass of the dense (fully connected) layer.

        This method performs a linear transformation of the input, applies
        the specified activation function, and optionally applies dropout
        during training. Supports mixed precision.

        Args:
            A_prev (Tensor): Input tensor of shape (N, input_features).
            training (bool, optional): Indicates whether the layer is in
                training mode. Dropout is applied only during training.

        Returns:
            Tensor: Output tensor of shape (N, nodes) after linear 
                transformation, activation, and optional dropout.
        """
        from LunarLearn.activations import get_activation
        from LunarLearn.regularizers import dropout

        if self.W is None or self.b is None:
            self.initialize((A_prev.shape[1],))
            
        W = self.W.to_compute()
        b = self.b.to_compute()

        if self.transpose_weight:
            Z = ops.matmul(A_prev, W.T) + b
        else:
            Z = ops.matmul(A_prev, W) + b

        # Activation
        activation = get_activation(self.activation)
        A = activation(Z)

        # Dropout (only during training)
        A = dropout(A, self.keep_prob, training=self.training)

        return A