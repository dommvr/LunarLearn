import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.nn.transformer.attention import ScaledDotProductAttention
from LunarLearn.core import Tensor, Parameter, ops
from LunarLearn.nn.transformer.utils.positional_encoding import apply_rope
from LunarLearn.train.finetuning import LoRAParameter

xp = backend.xp

class MultiHeadAttention(BaseLayer):
    def __init__(self, d_model, num_heads, n_kv_heads=None, attention=ScaledDotProductAttention, pos_mode="sinusoidal", keep_prob=1.0):
        super().__init__(trainable=True)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads or num_heads
        self.d_k = d_model // num_heads
        self.d_kv = d_model // self.n_kv_heads
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
        Wk = xp.random.randn(self.d_model, self.n_kv_heads * self.d_k) * scale
        Wv = xp.random.randn(self.d_model, self.n_kv_heads * self.d_k) * scale
        Wo = xp.random.randn(self.d_model, self.d_model) * scale

        self.Wq = Parameter(Wq, requires_grad=True)
        self.Wk = Parameter(Wk, requires_grad=True)
        self.Wv = Parameter(Wv, requires_grad=True)
        self.Wo = Parameter(Wo, requires_grad=True)

        self.output_shape = input_shape

    def _split_heads(self, x: Tensor, is_kv: bool = False) -> Tensor:
        """Split last dim into (num_heads, d_k) and transpose to (B, heads, L, d_k)."""
        B, L, D = x.shape
        head_dim = self.d_kv if is_kv else self.d_k
        n_heads = self.n_kv_heads if is_kv else self.num_heads
        x = x.reshape(B, L, n_heads, head_dim)
        return ops.transpose(x, (0, 2, 1, 3))
    
    def _repeat_kv(self, x: Tensor) -> Tensor:
        """Repeat KV heads to match num_heads."""
        if self.n_kv_heads == self.num_heads:
            return x
        B, H_kv, L, D = x.shape
        repeat_times = self.num_heads // self.n_kv_heads
        x = ops.repeat(x, repeat_times, axis=1)  # (B, H, L, D)
        return x

    def _combine_heads(self, x: Tensor) -> Tensor:
        """Inverse of _split_heads: (B, heads, L, d_k) → (B, L, D)."""
        B, H, L, Dk = x.shape
        x = ops.transpose(x, (0, 2, 1, 3))
        return x.reshape(B, L, H * Dk)

    def forward(self, x: Tensor, mask=None, context=None, cache=None, use_cache=False) -> Tensor:
        """
        Args:
            x: (B, L, D_model) - query input
            mask: optional attention mask
            context: optional encoder output for cross-attention (default: self-attn)
        """
        if self.Wq is None:
            self.initialize(x.shape[1:])

        Q_in = x
        K_in = context if context is not None else x
        V_in = K_in

        # Linear projections
        if isinstance(self.Wq, LoRAParameter):
            Q = self.Wq.forward(Q_in)
        else:
            Q = ops.matmul(Q_in, self.Wq.to_compute())

        if isinstance(self.Wk, LoRAParameter):
            K = self.Wk.forward(K_in)
        else:
            K = ops.matmul(K_in, self.Wk.to_compute())

        if isinstance(self.Wv, LoRAParameter):
            V = self.Wv.forward(V_in)
        else:
            V = ops.matmul(V_in, self.Wv.to_compute())

        # Split heads
        Q = self._split_heads(Q)
        K = self._split_heads(K, is_kv=True)
        V = self._split_heads(V, is_kv=True)

        if cache is not None:
            past_K, past_V = cache
            K = ops.concatenate([past_K, K], axis=2)
            V = ops.concatenate([past_V, V], axis=2)

        # Repeat KV heads to match Q
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # Apply rope if enabled
        start_pos = past_K.shape[2] if cache is not None else 0
        if self.pos_mode == "rotary":
            Q, K = apply_rope(Q, K, start_pos=start_pos)

        # Attention per head
        attn_out, attn = self.attention(Q, K, V, mask=mask, pos_mode=self.pos_mode)

        # Combine heads
        concat = self._combine_heads(attn_out)

        # Final linear projection
        if isinstance(self.Wo, LoRAParameter):
            out = self.Wo.forward(concat)
        else:
            out = ops.matmul(concat, self.Wo.to_compute())

        # Optional dropout
        out = ops.dropout(out, self.keep_prob, self.training)

        new_cache = (K, V) if use_cache else None

        return out, attn, new_cache
