import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp

class ScaledDotProductAttention(BaseLayer):
    def __init__(self, keep_prob=1.0):
        super().__init__(trainable=False)
        self.keep_prob = keep_prob

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask=None) -> Tensor:
        from LunarLearn.regularizers import dropout

        d_k = Q.shape[-1]
        scale = 1.0 / xp.sqrt(d_k)

        # Compute raw attention scores
        scores = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2))) * scale

        # Apply mask (if any)
        if mask is not None:
            scores += (mask * -1e9)

        # Softmax over last dimension (attention weights)
        attn = ops.softmax(scores, axis=-1)

        # Optional dropout for regularization
        attn = dropout(attn, self.keep_prob, self.training)

        # Weighted sum of values
        output = ops.matmul(attn, V)

        return output, attn

