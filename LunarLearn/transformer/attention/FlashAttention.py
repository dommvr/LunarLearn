import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp

class FlashAttention(BaseLayer):
    def __init__(self, keep_prob=1.0):
        super().__init__(trainable=False)
        self.keep_prob = keep_prob

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None) -> Tensor:
        from LunarLearn.regularizers import dropout

        d_k = Q.shape[-1]
        scale = 1.0 / xp.sqrt(d_k)

        # Compute scaled attention scores
        scores = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2))) * scale

        # Apply mask if any (e.g. causal or padding)
        if mask is not None:
            scores += mask * -1e9

        # Numerically stable softmax: subtract max before exp
        max_scores = ops.max(scores, axis=-1, keepdims=True)
        scores_exp = ops.exp(scores - max_scores)
        denom = ops.sum(scores_exp, axis=-1, keepdims=True)
        attn = scores_exp / denom

        attn = dropout(attn, self.keep_prob, self.training)
        output = ops.matmul(attn, V)

        return output, attn