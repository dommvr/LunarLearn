import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp

class LocalAttention(BaseLayer):
    def __init__(self, window_size=64, keep_prob=1.0):
        super().__init__(trainable=False)
        self.window_size = window_size
        self.keep_prob = keep_prob

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None) -> Tensor:
        from LunarLearn.regularizers import dropout

        B, H, L, D = Q.shape
        scale = 1.0 / xp.sqrt(D)

        output = []
        for i in range(L):
            # Local slice boundaries
            start = max(0, i - self.window_size)
            end = min(L, i + self.window_size + 1)

            Qi = Q[:, :, i:i+1, :]             # (B,H,1,D)
            Ki = K[:, :, start:end, :]         # (B,H,W,D)
            Vi = V[:, :, start:end, :]         # (B,H,W,D)

            scores = ops.matmul(Qi, ops.transpose(Ki, (0, 1, 3, 2))) * scale  # (B,H,1,W)
            if mask is not None:
                scores += mask[:, :, i:i+1, start:end] * -1e9

            attn = ops.softmax(scores, axis=-1)
            attn = dropout(attn, self.keep_prob, self.training)
            out = ops.matmul(attn, Vi)         # (B,H,1,D)
            output.append(out)

        output = ops.concatenate(output, axis=2)
        return output, None
