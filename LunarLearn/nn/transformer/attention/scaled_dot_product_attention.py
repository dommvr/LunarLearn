import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops
from LunarLearn.nn.transformer.utils.positional_encoding import get_alibi_bias

xp = backend.xp

class ScaledDotProductAttention(BaseLayer):
    def __init__(self, keep_prob=1.0):
        super().__init__(trainable=False)
        self.keep_prob = keep_prob
        self.rel_bias = None

    def initialize(self, n_heads, max_len):
        self.rel_bias = Parameter(xp.zeros((n_heads, max_len, max_len)), requires_grad=True)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None, pos_mode=None) -> Tensor:
        n_heads, max_len, d_k = Q.shape[1], Q.shape[2], Q.shape[-1]

        # Lazy init for relative bias
        if pos_mode == "relative" and self.rel_bias is None:
            self.initialize(n_heads, max_len)

        scale = 1.0 / xp.sqrt(d_k)

        # Compute raw attention scores
        scores = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2))) * scale

        # Add position-based bias
        if pos_mode == "alibi":
            alibi_bias = get_alibi_bias(Q, K)
            scores += alibi_bias
        elif pos_mode == "relative":
            scores += self.rel_bias[:, :max_len, :max_len]

        # Apply mask (if any)
        if mask is not None:
            scores += (1.0 - mask) * -1e9

        # Softmax over last dimension (attention weights)
        attn = ops.softmax(scores, axis=-1)

        # Optional dropout for regularization
        attn = ops.dropout(attn, self.keep_prob, self.training)

        # Weighted sum of values
        output = ops.matmul(attn, V)

        return output, attn

