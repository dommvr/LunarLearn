import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

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

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None) -> Tensor:
        from LunarLearn.regularizers import dropout

        Qf = self._phi(Q)
        Kf = self._phi(K)

        if mask is not None:
            Kf = Kf * (1.0 - mask)

        # Compute key-value summary once
        KV = ops.matmul(ops.transpose(Kf, (0, 1, 3, 2)), V)  # (B,H,D,D)
        denom = ops.matmul(Qf, ops.sum(Kf, axis=2, keepdims=True)) + 1e-6
        numer = ops.matmul(Qf, KV)
        out = numer / denom

        out = dropout(out, self.keep_prob, self.training)
        return out, None
