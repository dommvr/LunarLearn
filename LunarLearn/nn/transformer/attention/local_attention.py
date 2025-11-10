import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops
from LunarLearn.nn.transformer.utils.positional_encoding import get_alibi_bias

xp = backend.xp

class LocalAttention(BaseLayer):
    def __init__(self, window_size=64, keep_prob=1.0):
        super().__init__(trainable=False)
        self.window_size = window_size
        self.keep_prob = keep_prob
        self.rel_bias = None

    def initialize(self, n_heads, max_len):
        self.rel_bias = Parameter(xp.zeros((n_heads, max_len, max_len)), requires_grad=True)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None, pos_mode=None) -> Tensor:
        B, H, L, D = Q.shape

        # Lazy init for relative bias
        if pos_mode == "relative" and self.rel_bias is None:
            self.initialize(H, L)

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

            # Add position-based bias
            if pos_mode == "alibi":
                alibi_bias = get_alibi_bias(Q, K)
                scores += alibi_bias[:, i:i+1, start:end]
            elif pos_mode == "relative":
                scores += self.rel_bias[:, i:i+1, start:end]

            if mask is not None:
                scores += (1.0 - mask[:, :, i:i+1, start:end]) * -1e9

            attn = ops.softmax(scores, axis=-1)
            attn = ops.dropout(attn, self.keep_prob, self.training)
            out = ops.matmul(attn, Vi)         # (B,H,1,D)
            output.append(out)

        output = ops.concatenate(output, axis=2)
        return output, None
