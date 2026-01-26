import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, ConvLSTMCell
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE


class ConvLSTM(BaseLayer):
    def __init__(self,
                 hidden_channels: int,
                 kernel_size: int | tuple = 3,
                 padding: int | str = "same",
                 bias: bool = True,
                 return_sequences: bool = False,
                 stateful: bool = False,
                 recurrent_keep_prob: float = 1.0):
        super().__init__(trainable=True)

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        self.return_sequences = return_sequences
        self.stateful = stateful
        self.recurrent_keep_prob = recurrent_keep_prob

        self.cell = ConvLSTMCell(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

        # persistent states if stateful
        self.h_state = None
        self.c_state = None

        # caches (if you later want BPTT tricks)
        self.h_cache = None
        self.x_cache = None

    def reset_state(self, batch_size=None, spatial_shape=None, dtype=DTYPE):
        """
        Reset internal state (h_state, c_state).

        If batch_size and spatial_shape provided, pre-allocate state.
        """
        if batch_size is None or spatial_shape is None:
            self.h_state = None
            self.c_state = None
            return

        C_h = self.hidden_channels
        H, W = spatial_shape
        h = xp.zeros((batch_size, C_h, H, W), dtype=dtype)
        c = xp.zeros((batch_size, C_h, H, W), dtype=dtype)
        self.h_state = Tensor(h, requires_grad=False, dtype=dtype)
        self.c_state = Tensor(c, requires_grad=False, dtype=dtype)

    def initialize(self, input_shape):
        """
        input_shape: (T, C_in, H, W) (no batch dim)
        """
        T, C_in, H, W = input_shape

        # cell output: (C_h, H, W)
        self.cell.initialize((C_in, H, W))

        if self.return_sequences:
            self.output_shape = (T, self.hidden_channels, H, W)
        else:
            self.output_shape = (self.hidden_channels, H, W)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, C_in, H, W)
        """
        from LunarLearn.core.tensor import RecurrentDropout

        B, T, C_in, H, W = x.shape

        # initialize or reuse states
        if self.stateful and self.h_state is not None and self.c_state is not None:
            h = self.h_state
            c = self.c_state
        else:
            h = Tensor(xp.zeros((B, self.hidden_channels, H, W), dtype=x.dtype),
                       requires_grad=False, dtype=x.dtype)
            c = Tensor(xp.zeros((B, self.hidden_channels, H, W), dtype=x.dtype),
                       requires_grad=False, dtype=x.dtype)

        hs = []
        self.x_cache, self.h_cache = x, []

        # Recurrent dropout mask over h (one mask per sequence)
        # If your RecurrentDropout is generic over shape, this will work.
        dropout = RecurrentDropout(
            (B, self.hidden_channels, H, W),
            self.recurrent_keep_prob,
            self.training
        )

        for t in range(T):
            x_t = x[:, t, :, :, :]  # (B, C_in, H, W)

            # recurrent dropout on hidden state
            h = dropout(h)

            # ConvLSTM step
            h, c = self.cell.forward(x_t, h, c)

            h_cache_entry = h.astype(DTYPE, copy=False)
            self.h_cache.append(h_cache_entry)
            if self.return_sequences:
                hs.append(h_cache_entry)

        # handle stateful
        if self.stateful:
            # detach to avoid crazy graph over epochs
            self.h_state = h.detach()
            self.c_state = c.detach()

        # output formatting
        if self.return_sequences:
            # (T, B, C, H, W) -> (B, T, C, H, W)
            out = ops.stack(hs, axis=1)
        else:
            out = h  # last hidden state (B, C_h, H, W)

        return out