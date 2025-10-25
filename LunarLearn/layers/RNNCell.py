import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

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
    def __init__(self, hidden_size, activation="tanh", w_init="auto", uniform=False, gain=1):
        super().__init__(trainable=True)
        self.hidden_size = hidden_size
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain

        self.Wxh = None
        self.Whh = None
        self.bh = None

    def initialize(self, input_shape):
        from LunarLearn.initializations import initialize_weights

        n_in = input_shape[-1]
        Wxh, bh = initialize_weights(
            (n_in, self.hidden_size),
            (1, self.hidden_size),
            self.w_init, self.activation,
            self.uniform, self.gain
        )
        Whh, _ = initialize_weights(
            (self.hidden_size, self.hidden_size),
            (1, self.hidden_size),
            self.w_init, self.activation,
            self.uniform, self.gain
        )

        self.Wxh = Parameter(Wxh, requires_grad=True)
        self.Whh = Parameter(Whh, requires_grad=True)
        self.bh = Parameter(bh, requires_grad=True)

        self._apply_param_settings()

        self.output_shape = (self.hidden_size,)

    def forward(self, x_t: Tensor, h_prev: Tensor) -> Tensor:
        from LunarLearn.activations import get_activation

        Wxh = self.Wxh.to_compute()
        Whh = self.Whh.to_compute()
        bh = self.bh.to_compute()

        z = ops.matmul(x_t, Wxh) + ops.matmul(h_prev, Whh) + bh

        activation = get_activation(self.activation)
        A = activation(z)

        return A