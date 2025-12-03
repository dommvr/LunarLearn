import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, Conv2D
from LunarLearn.core import Tensor, ops

xp = backend.xp


class ConvLSTMCell(BaseLayer):
    def __init__(self,
                 hidden_channels: int,
                 kernel_size: int | tuple = 3,
                 padding: int | str = "same",
                 bias: bool = True):
        super().__init__(trainable=True)

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        # Conv over [x_t, h_prev]
        self.gate_conv = Conv2D(
            filters=4*hidden_channels,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            bias=bias
        )

    def initialize(self, input_shape):
        """
        input_shape: (C_in, H, W) — we don't need anything special here,
        Conv2D will lazy-initialize on first forward.
        """
        C_in, H, W = input_shape
        self.output_shape = (self.hidden_channels, H, W)

    def _init_state(self, x_t: Tensor) -> tuple[Tensor, Tensor]:
        B, _, H, W = x_t.shape
        h = xp.zeros((B, self.hidden_channels, H, W), dtype=x_t.dtype)
        c = xp.zeros((B, self.hidden_channels, H, W), dtype=x_t.dtype)
        h = Tensor(h, requires_grad=True, dtype=x_t.dtype)
        c = Tensor(c, requires_grad=True, dtype=x_t.dtype)
        return h, c

    def forward(self,
                x_t: Tensor,
                h_prev: Tensor | None = None,
                c_prev: Tensor | None = None) -> tuple[Tensor, Tensor]:

        if h_prev is None or c_prev is None:
            h_prev, c_prev = self._init_state(x_t)

        # concat along channels: [x_t, h_prev]
        combined = ops.concatenate([x_t, h_prev], axis=1)  # (B, C_in + C_h, H, W)

        gates = self.gate_conv(combined)  # (B, 4*C_h, H, W)

        C_h = self.hidden_channels
        i = gates[:, 0:C_h, :, :]
        f = gates[:, C_h:2 * C_h, :, :]
        o = gates[:, 2 * C_h:3 * C_h, :, :]
        g = gates[:, 3 * C_h:4 * C_h, :, :]

        i = ops.sigmoid(i)
        f = ops.sigmoid(f)
        o = ops.sigmoid(o)
        g = ops.tanh(g)

        c_new = f * c_prev + i * g
        h_new = o * ops.tanh(c_new)

        return h_new, c_new