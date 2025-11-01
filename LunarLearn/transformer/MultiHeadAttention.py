import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.transformer.attention import ScaledDotProductAttention
from LunarLearn.transformer.utils.positional_encoding import apply_rope, get_alibi_bias
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter
from LunarLearn.tensor import ops

xp = backend.xp

class MultiHeadAttention(BaseLayer):
    def __init__(self, d_model, num_heads, attention=ScaledDotProductAttention, pos_mode="sinusoidal", keep_prob=1.0):
        super().__init__(trainable=True)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.keep_prob = keep_prob

        # Projection matrices for Q, K, V, and output
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None

        self.attention = attention(keep_prob=keep_prob)
        self.pos_mode = pos_mode

    def initialize(self, input_shape):
        scale = 1 / xp.sqrt(self.d_model)
        Wq = xp.random.randn(self.d_model, self.d_model) * scale
        Wk = xp.random.randn(self.d_model, self.d_model) * scale
        Wv = xp.random.randn(self.d_model, self.d_model) * scale
        Wo = xp.random.randn(self.d_model, self.d_model) * scale

        self.Wq = Parameter(Wq, requires_grad=True)
        self.Wk = Parameter(Wk, requires_grad=True)
        self.Wv = Parameter(Wv, requires_grad=True)
        self.Wo = Parameter(Wo, requires_grad=True)

        self.output_shape = input_shape

    def _split_heads(self, x: Tensor) -> Tensor:
        """Split last dim into (num_heads, d_k) and transpose to (B, heads, L, d_k)."""
        B, L, D = x.shape
        x = x.reshape(B, L, self.num_heads, self.d_k)
        return ops.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: Tensor) -> Tensor:
        """Inverse of _split_heads: (B, heads, L, d_k) → (B, L, D)."""
        B, H, L, Dk = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        return x.reshape(B, L, H * Dk)

    def forward(self, x: Tensor, mask=None, context=None) -> Tensor:
        """
        Args:
            x: (B, L, D_model) - query input
            mask: optional attention mask
            context: optional encoder output for cross-attention (default: self-attn)
        """
        from LunarLearn.regularizers import dropout
        if self.Wq is None:
            self.initialize(x.shape[1:])

        Q_in = x
        K_in = context if context is not None else x
        V_in = K_in

        # Linear projections
        Q = ops.matmul(Q_in, self.Wq.to_compute())
        K = ops.matmul(K_in, self.Wk.to_compute())
        V = ops.matmul(V_in, self.Wv.to_compute())

        # Split heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Attention per head
        attn_out, attn = self.attention(Q, K, V, mask=mask, pos_mode=self.pos_mode)

        # Combine heads
        concat = self._combine_heads(attn_out)

        # Final linear projection
        out = ops.matmul(concat, self.Wo.to_compute())

        # Optional dropout
        out = dropout(out, self.keep_prob, self.training)

        return out, attn
