import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Parameter, ops

xp = backend.xp
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
    def __init__(self, hidden_size, activation="tanh", w_init="auto", uniform=False, gain=1, zero_bias=True, bias=True):
        super().__init__(trainable=True)
        self.hidden_size = hidden_size
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain
        self.zero_bias = zero_bias
        self.bias = bias

    def initialize(self, input_shape):
        from LunarLearn.nn.initializations import initialize_weights

        n_in = input_shape[-1]
        hs = self.hidden_size

        # Input weights + input biases (bias_ih)
        self.Wz, self.bz_ih = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Wr, self.br_ih = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain, 
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Wh, self.bh_ih = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )

        # Recurrent weights + recurrent biases (bias_hh)
        self.Uz, self.bz_hh = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Ur, self.br_hh = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Uh, self.bh_hh = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )

        # Wrap parameters in Tensors (skip None biases when bias=False)
        for name in [
            "Wz", "Wr", "Wh",
            "Uz", "Ur", "Uh",
            "bz_ih", "br_ih", "bh_ih",
            "bz_hh", "br_hh", "bh_hh"
        ]:
            arr = getattr(self, name, None)
            if arr is not None:
                setattr(self, name, Parameter(arr, requires_grad=True))

        self._apply_param_settings()
        self.output_shape = (hs,)

    def forward(self, x_t, h_prev):
        Wz, Wr, Wh = self.Wz.to_compute(), self.Wr.to_compute(), self.Wh.to_compute()
        Uz, Ur, Uh = self.Uz.to_compute(), self.Ur.to_compute(), self.Uh.to_compute()

        if self.bias:
            bz_ih, br_ih, bh_ih = self.bz_ih.to_compute(), self.br_ih.to_compute(), self.bh_ih.to_compute()
            bz_hh, br_hh, bh_hh = self.bz_hh.to_compute(), self.br_hh.to_compute(), self.bh_hh.to_compute()
        else:
            # safer than raw Python 0 if your backend gets picky about dtype/device
            z0 = xp.array(0, dtype=DTYPE)
            bz_ih = br_ih = bh_ih = z0
            bz_hh = br_hh = bh_hh = z0

        z_t = ops.sigmoid(ops.matmul(x_t, Wz) + bz_ih + ops.matmul(h_prev, Uz) + bz_hh)
        r_t = ops.sigmoid(ops.matmul(x_t, Wr) + br_ih + ops.matmul(h_prev, Ur) + br_hh)
        h_tilde = ops.tanh(ops.matmul(x_t, Wh) + bh_ih + ops.matmul(r_t * h_prev, Uh) + bh_hh)

        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t