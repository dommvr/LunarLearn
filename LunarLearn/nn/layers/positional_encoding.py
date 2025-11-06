import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter

xp = backend.xp

class PositionalEncoding(BaseLayer):
    def __init__(self, emb_dim, max_len, mode="sinusoidal"):
        super().__init__(trainable=True if mode == "learnable" else False)
        self.P = None
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.mode = mode

        modes = ["sinusoidal", "learnable", "rotary", "alibi", "relative", "none"]

    def initialize(self, input_shape):
        if self.mode == "learnable":
            scale = 1 / xp.sqrt(self.emb_dim)
            P = xp.random.randn(self.max_len, self.emb_dim) * scale
            self.P = Parameter(P, requires_grad=True)
        self.output_shape = input_shape

    def _get_angles(self):
        pos = xp.arange(self.max_len)[:, xp.newaxis]
        i = xp.arange(self.emb_dim)[xp.newaxis, :]
        angles = pos / (10000.0 ** (2 * (i // 2) / self.emb_dim))
        return angles
    
    def forward(self, x: Tensor) -> Tensor:
        if self.output_shape is None:
            self.initialize(x.shape[1:])

        seq_len = x.shape[1]

        if self.mode == "sinusoidal":
            angle_rads = self._get_angles()
            angle_rads[:, 0::2] = xp.sin(angle_rads[:, 0::2])
            angle_rads[:, 1::2] = xp.cos(angle_rads[:, 1::2])
            pos_encoding = angle_rads[xp.newaxis, :seq_len, :]
        elif self.mode == "learnable":
            P = self.P.to_compute()
            pos_encoding = P[:seq_len][xp.newaxis, :, :]
        else:
            pos_encoding = None

        if pos_encoding is not None:
            x += pos_encoding
        
        return x