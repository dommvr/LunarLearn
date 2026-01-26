import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION


class LSTMCell(BaseLayer):
    """
    Long Short-Term Memory (LSTM) cell.

    Computes hidden and cell states using gating:
        i_t = sigmoid(x_t @ Wi + h_{t-1} @ Ui + bi)
        f_t = sigmoid(x_t @ Wf + h_{t-1} @ Uf + bf)
        o_t = sigmoid(x_t @ Wo + h_{t-1} @ Uo + bo)
        g_t = tanh(x_t @ Wg + h_{t-1} @ Ug + bg)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    Handles mixed precision and autograd, and is designed to work within
    the `RecurrentBase` class (via the `LSTM` layer).

    Args:
        hidden_size (int): Number of hidden units.
        activation (str, optional): Activation function for candidate state (`g_t`).
            Defaults to `"tanh"`.
        w_init (str, optional): Weight initialization method. Defaults to `"auto"`.
        uniform (bool, optional): Whether to use uniform weight initialization.
            Defaults to False.
        gain (float, optional): Scaling gain for initialization. Defaults to 1.

    Returns:
        Tuple[Tensor, Tensor]: `(h_t, c_t)` — updated hidden and cell states.
    """
    def __init__(self, hidden_size, activation="tanh", w_init="auto", uniform=False, gain=1, zero_bias=True, forget_bias=1.0, bias=True):
        super().__init__(trainable=True)
        self.hidden_size = hidden_size
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain
        self.zero_bias = zero_bias
        self.forget_bias = forget_bias / 2
        self.bias = bias

    def initialize(self, input_shape):
        from LunarLearn.nn.initializations import initialize_weights

        n_in = input_shape[-1]
        hs = self.hidden_size

        # Input gates
        self.Wi, self.bi = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            self.uniform, self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Wf, self.bf = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            self.uniform, self.gain,
            bias_value=self.forget_bias, bias=self.bias
        )
        self.Wo, self.bo = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            self.uniform, self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Wg, self.bg = initialize_weights(
            (n_in, hs), (1, hs), self.w_init, self.activation,
            self.uniform, self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )

        # Recurrent gates
        self.Ui, self.ubi = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Uf, self.ubf = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            bias_value=self.forget_bias, bias=self.bias
        )
        self.Uo, self.ubo = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )
        self.Ug, self.ubg = initialize_weights(
            (hs, hs), (1, hs), self.w_init, self.activation,
            uniform=self.uniform, gain=self.gain,
            zero_bias=self.zero_bias, bias=self.bias
        )

        for name in ["Wi", "Wf", "Wo", "Wg", "Ui", "Uf", "Uo", "Ug",
                     "bi", "bf", "bo", "bg", "ubi", "ubf", "ubo", "ubg"]:
            arr = getattr(self, name, None)
            if arr is not None:
                setattr(self, name, Parameter(arr, requires_grad=True))

        self._apply_param_settings()

        self.output_shape = (self.hidden_size,)

    def forward(self, x_t, h_prev_c_prev):
        h_prev, c_prev = h_prev_c_prev

        Wi, Wf, Wo, Wg = self.Wi.to_compute(), self.Wf.to_compute(), self.Wo.to_compute(), self.Wg.to_compute()
        Ui, Uf, Uo, Ug = self.Ui.to_compute(), self.Uf.to_compute(), self.Uo.to_compute(), self.Ug.to_compute()
        
        if self.bias:
            bi, bf, bo, bg = self.bi.to_compute(), self.bf.to_compute(), self.bo.to_compute(), self.bg.to_compute()
            ubi, ubf, ubo, ubg = self.ubi.to_compute(), self.ubf.to_compute(), self.ubo.to_compute(), self.ubg.to_compute()
        else:
            z0 = xp.array(0, dtype=DTYPE)
            bi = bf = bo = bg = z0
            ubi = ubf = ubo = ubg = z0

        i_t = ops.sigmoid(ops.matmul(x_t, Wi) + ops.matmul(h_prev, Ui) + bi + ubi)
        f_t = ops.sigmoid(ops.matmul(x_t, Wf) + ops.matmul(h_prev, Uf) + bf + ubf)
        o_t = ops.sigmoid(ops.matmul(x_t, Wo) + ops.matmul(h_prev, Uo) + bo + ubo)
        g_t = ops.tanh(ops.matmul(x_t, Wg) + ops.matmul(h_prev, Ug) + bg + ubg)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * ops.tanh(c_t)

        return h_t, c_t