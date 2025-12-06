from LunarLearn.nn import ModuleList
from LunarLearn.nn.layers import BaseLayer, Dense, LayerNorm
from LunarLearn.nn.transformer.attention import WindowAttention
from LunarLearn.core import Tensor, ops


class SwinBlock(BaseLayer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, norm_layer=LayerNorm):
        super().__init__(trainable=True)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer()
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = norm_layer()
        self.mlp = ModuleList([Dense(dim * 4, activation="gelu"), Dense(dim)])

    def _window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
        return x

    def _window_reverse(windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
        return x

    def forward(self, x: Tensor, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x.reshape(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = ops.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        x_windows = self._window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size ** 2, C)

        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = ops.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        x = x.reshape(B, H * W, C)
        x = shortcut + x
        out = x + self.mlp(self.norm2(x))
        return out
