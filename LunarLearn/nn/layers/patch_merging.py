from LunarLearn.nn.layers import BaseLayer, Dense, LayerNorm
from LunarLearn.core import Tensor, ops


class PatchMerging(BaseLayer):
    def __init__(self, input_resolution, dim):
        super().__init__(trainable=True)
        self.input_resolution = input_resolution if isinstance(input_resolution, tuple) else (input_resolution, input_resolution)
        self.dim = dim
        self.reduction = Dense(2 * dim, bias=False)
        self.norm = LayerNorm()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, L, C = x.shape
        x = x.reshape(B, H, W, C)

        pad_h = (self.input_resolution[0] - H) % 2
        pad_w = (self.input_resolution[1] - W) % 2
        if pad_h or pad_w:
            x = ops.pad(x, (0, 0, 0, pad_h, 0, pad_w, 0, 0))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)
        x = x.reshape(B, -1, 4 * C)

        x = self.norm(x)
        out = self.reduction(x)
        return out