import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

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
    def __init__(self, hidden_size, activation="tanh", w_init="auto", uniform=False, gain=1):
        super().__init__(trainable=True)
        self.hidden_size = hidden_size
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain

    def initialize(self, input_shape):
        from LunarLearn.initializations import initialize_weights

        n_in = input_shape[-1]

        def init_pair():
            return initialize_weights((n_in, self.hidden_size),
                                      (1, self.hidden_size),
                                      self.w_init, self.activation,
                                      self.uniform, self.gain)

        # Input gates
        self.Wi, self.bi = init_pair()
        self.Wf, self.bf = init_pair()
        self.Wo, self.bo = init_pair()
        self.Wg, self.bg = init_pair()

        # Recurrent gates
        self.Ui, _ = initialize_weights((self.hidden_size, self.hidden_size),
                                        (1, self.hidden_size),
                                        self.w_init, self.activation,
                                        self.uniform, self.gain)
        self.Uf, _ = initialize_weights((self.hidden_size, self.hidden_size),
                                        (1, self.hidden_size),
                                        self.w_init, self.activation,
                                        self.uniform, self.gain)
        self.Uo, _ = initialize_weights((self.hidden_size, self.hidden_size),
                                        (1, self.hidden_size),
                                        self.w_init, self.activation,
                                        self.uniform, self.gain)
        self.Ug, _ = initialize_weights((self.hidden_size, self.hidden_size),
                                        (1, self.hidden_size),
                                        self.w_init, self.activation,
                                        self.uniform, self.gain)

        for name in ["Wi", "Wf", "Wo", "Wg", "Ui", "Uf", "Uo", "Ug", "bi", "bf", "bo", "bg"]:
            arr = getattr(self, name)
            setattr(self, name, Parameter(arr, requires_grad=True))

        self.output_shape = (self.hidden_size,)

    def forward(self, x_t, h_prev_c_prev):
        h_prev, c_prev = h_prev_c_prev

        Wi, Ui, bi = self.Wi.to_compute(), self.Ui.to_compute(), self.bi.to_compute()
        Wf, Uf, bf = self.Wf.to_compute(), self.Uf.to_compute(), self.bf.to_compute()
        Wo, Uo, bo = self.Wo.to_compute(), self.Uo.to_compute(), self.bo.to_compute()
        Wg, Ug, bg = self.Wg.to_compute(), self.Ug.to_compute(), self.bg.to_compute()

        i_t = ops.sigmoid(ops.matmul(x_t, Wi) + ops.matmul(h_prev, Ui) + bi)
        f_t = ops.sigmoid(ops.matmul(x_t, Wf) + ops.matmul(h_prev, Uf) + bf)
        o_t = ops.sigmoid(ops.matmul(x_t, Wo) + ops.matmul(h_prev, Uo) + bo)
        g_t = ops.tanh(ops.matmul(x_t, Wg) + ops.matmul(h_prev, Ug) + bg)

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * ops.tanh(c_t)

        return h_t, c_t