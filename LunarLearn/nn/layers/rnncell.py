import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class RNNCell(BaseLayer):
    """
    Vanilla Recurrent Neural Network (RNN) cell.

    Computes the hidden state update at a single timestep:
        h_t = activation(x_t @ Wxh + h_{t-1} @ Whh + b)

    Supports mixed precision and autograd, and is designed to be unrolled
    inside a `RecurrentBase`-derived layer (e.g., `RNN`, `GRU`, or `LSTM`).

    Args:
        hidden_size (int): Number of hidden units.
        activation (str, optional): Activation function to use (e.g. `"tanh"`, `"relu"`).
            Defaults to `"tanh"`.
        w_init (str, optional): Weight initialization method (e.g. `"xavier"`, `"he"`, `"auto"`).
            Defaults to `"auto"`.
        uniform (bool, optional): Whether to use uniform distribution for initialization.
            Defaults to False.
        gain (float, optional): Scaling gain for initialization. Defaults to 1.

    Returns:
        Tensor: Hidden state at current timestep of shape `(batch, hidden_size)`.
    """
    def __init__(self, hidden_size, activation="tanh", w_init="auto", uniform=False, gain=1, zero_bias=True, bias=True):
        super().__init__(trainable=True)
        self.hidden_size = hidden_size
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain
        self.zero_bias = zero_bias
        self.bias = bias

        self.Wxh = None
        self.Whh = None
        self.bxh = None
        self.bhh = None

    def initialize(self, input_shape):
        from LunarLearn.nn.initializations import initialize_weights

        n_in = input_shape[-1]
        hs = self.hidden_size

        Wxh, bxh = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        Whh, bhh = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )

        self.Wxh = Parameter(Wxh, requires_grad=True)
        self.Whh = Parameter(Whh, requires_grad=True)
        if self.bias:
            self.bxh = Parameter(bxh, requires_grad=True)
            self.bhh = Parameter(bhh, requires_grad=True)

        self._apply_param_settings()

        self.output_shape = (self.hidden_size,)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        from LunarLearn.nn.activations import get_activation

        Wxh = self.Wxh.to_compute()
        Whh = self.Whh.to_compute()
        if self.bias:
            bxh = self.bxh.to_compute()
            bhh = self.bhh.to_compute()
        else:
            z0 = xp.array(0, dtype=DTYPE)
            bxh = bhh = z0

        z = ops.matmul(x_t, Wxh) + ops.matmul(h_prev, Whh) + bxh + bhh

        activation = get_activation(self.activation)
        A = activation(z)

        return A