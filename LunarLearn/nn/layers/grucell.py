import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Parameter, ops

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION


class GRUCell(BaseLayer):
    """
    Gated Recurrent Unit (GRU) cell.

    Computes the hidden state using gating mechanisms:
        z_t = sigmoid(x_t @ Wz + h_{t-1} @ Uz + bz)
        r_t = sigmoid(x_t @ Wr + h_{t-1} @ Ur + br)
        h̃_t = tanh(x_t @ Wh + (r_t * h_{t-1}) @ Uh + bh)
        h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t

    Designed for efficient sequence modeling and implemented with
    autograd and mixed-precision support.

    Args:
        hidden_size (int): Number of hidden units.
        activation (str, optional): Activation function for candidate state (`h̃_t`).
            Defaults to `"tanh"`.
        w_init (str, optional): Weight initialization strategy. Defaults to `"auto"`.
        uniform (bool, optional): Whether to sample weights from a uniform distribution.
            Defaults to False.
        gain (float, optional): Scaling gain for initialization. Defaults to 1.

    Returns:
        Tensor: Updated hidden state `(batch, hidden_size)`.
    """
    def __init__(self, hidden_size, activation="tanh", w_init="auto", uniform=False, gain=1):
        super().__init__(trainable=True)
        self.hidden_size = hidden_size
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain

    def initialize(self, input_shape):
        from LunarLearn.nn.initializations import initialize_weights

        n_in = input_shape[-1]

        def init_pair():
            return initialize_weights((n_in, self.hidden_size),
                                      (1, self.hidden_size),
                                      self.w_init, self.activation,
                                      self.uniform, self.gain)

        self.Wz, self.bz = init_pair()
        self.Wr, self.br = init_pair()
        self.Wh, self.bh = init_pair()

        # Recurrent weights
        self.Uz, _ = initialize_weights((self.hidden_size, self.hidden_size),
                                        (1, self.hidden_size),
                                        self.w_init, self.activation,
                                        self.uniform, self.gain)
        self.Ur, _ = initialize_weights((self.hidden_size, self.hidden_size),
                                        (1, self.hidden_size),
                                        self.w_init, self.activation,
                                        self.uniform, self.gain)
        self.Uh, _ = initialize_weights((self.hidden_size, self.hidden_size),
                                        (1, self.hidden_size),
                                        self.w_init, self.activation,
                                        self.uniform, self.gain)

        # Wrap parameters in Tensors
        for name in ["Wz", "Wr", "Wh", "Uz", "Ur", "Uh", "bz", "br", "bh"]:
            arr = getattr(self, name)
            setattr(self, name, Parameter(arr, requires_grad=True))

        self._apply_param_settings()

        self.output_shape = (self.hidden_size,)

    def forward(self, x_t, h_prev):
        Wz, Uz, bz = self.Wz.to_compute(), self.Uz.to_compute(), self.bz.to_compute()
        Wr, Ur, br = self.Wr.to_compute(), self.Ur.to_compute(), self.br.to_compute()
        Wh, Uh, bh = self.Wh.to_compute(), self.Uh.to_compute(), self.bh.to_compute()

        z_t = ops.sigmoid(ops.matmul(x_t, Wz) + ops.matmul(h_prev, Uz) + bz)
        r_t = ops.sigmoid(ops.matmul(x_t, Wr) + ops.matmul(h_prev, Ur) + br)
        h_tilde = ops.tanh(ops.matmul(x_t, Wh) + ops.matmul(r_t * h_prev, Uh) + bh)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        return h_t