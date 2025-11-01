import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops
from LunarLearn.transformer.utils.positional_encoding import apply_rope

xp = backend.xp

class LinearAttention(BaseLayer):
    def __init__(self, feature_map="elu", keep_prob=1.0):
        super().__init__(trainable=False)
        self.keep_prob = keep_prob
        self.feature_map = feature_map

    def _phi(self, x: Tensor) -> Tensor:
        if self.feature_map == "elu":
            return ops.elu(x) + 1.0
        elif self.feature_map == "relu":
            return ops.relu(x)
        else:
            return x  # identity fallback

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None, pos_mode=None) -> Tensor:
        from LunarLearn.regularizers import dropout

        if pos_mode == "rotary":
            Q, K = apply_rope(Q, K)

        Qf = self._phi(Q)
        Kf = self._phi(K)

        # Apply mask (1 for keep, 0 for pad)
        if mask is not None:
            Kf = Kf * mask[:, None, :, None]
            V = V * mask[:, None, :, None]

        # Compute key-value summary once
        KV = ops.matmul(ops.transpose(Kf, (0, 1, 3, 2)), V)  # (B,H,D,D)
        inv_denom = 1.0 / (ops.matmul(Qf, ops.sum(Kf, axis=2, keepdims=True)) + 1e-6)
        numer = ops.matmul(Qf, KV)
        out = numer * inv_denom

        out = dropout(out, self.keep_prob, self.training)
        return out, None
