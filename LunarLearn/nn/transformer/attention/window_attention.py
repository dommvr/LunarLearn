import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, Dense
from LunarLearn.core import Tensor, Parameter, ops
from LunarLearn.nn.transformer.utils.positional_encoding import get_relative_position_index

xp = backend.xp


class WindowAttention(BaseLayer):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__(trainable=True)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.relative_bias_table = None
        self.coords = None

        self.qkv = Dense(dim * 3)
        self.proj = Dense(dim)

    def initialize(self, input_shape):
        p = xp.random.randn((2 * self.window_size - 1) ** 2, self.num_heads)
        self.relative_bias_table = Parameter(p, requires_grad=True)
        self.coords = get_relative_position_index(self.window_size, self.window_size)
        self.output_shape = (input_shape)

    def forward(self, x: Tensor, mask=None):
        B_, N, C = x.shape

        if self.relative_bias_table is None:
            self.initialize(x.shape)

        relative_bias_table = self.relative_bias_table.to_compute()
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale
        attn = ops.matmul(q, k.transpose(-2, -1))

        relative_bias = relative_bias_table[self.coords.reshape(-1)]  # Index flat
        relative_bias = relative_bias.reshape(self.window_size**2, self.window_size**2, -1)
        relative_bias = relative_bias.transpose(2, 0, 1)  # (nH, N, N)
        attn += relative_bias[None, :, :, :]

        if mask is not None:
            attn += (1.0 - mask) * -1e9

        attn = ops.softmax(attn, axis=-1)
        x = ops.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x
