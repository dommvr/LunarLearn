from LunarLearn.nn.transformer.attention import ScaledDotProductAttention
from LunarLearn.core import Tensor, ops
from LunarLearn.nn.transformer.utils.positional_encoding import apply_rope, get_alibi_bias

class CausalScaledDotProductAttention(ScaledDotProductAttention):
    def forward(self, Q, K, V, mask=None, pos_mode=None):
        n_heads, seq_len, d_k = Q.shape[1], Q.shape[2], Q.shape[-1]

        # === LAZY INIT: same as parent ===
        if pos_mode == "relative" and self.rel_bias is None:
            self.initialize(n_heads, seq_len)

        scale = 1.0 / xp.sqrt(d_k)

        # === ROTARY (optional) ===
        if pos_mode == "rotary":
            Q, K = apply_rope(Q, K)

        # === SCORES ===
        scores = ops.matmul(Q, ops.transpose(K, (0, 1, 3, 2))) * scale

        # === POSITION BIAS ===
        if pos_mode == "alibi":
            scores += get_alibi_bias(Q, K)
        elif pos_mode == "relative":
            scores += self.rel_bias[:, :seq_len, :seq_len]

        # === CAUSAL MASK (NEW) ===
        # Shape: (1, 1, seq_len, seq_len)
        causal_mask = ops.tril(ops.ones(seq_len, seq_len))
        causal_mask = ops.expand_dims(causal_mask, axis=[0, 1])  # (1,1,T,T)
        # Convert 0→-inf, 1→0
        causal_mask = (1.0 - causal_mask) * -1e9
        scores = scores + causal_mask

        # === USER MASK (optional padding) ===
        if mask is not None:
            # Assume mask: (B, 1, 1, T) or (B, T)
            mask = ops.expand_dims(mask, axis=[1, 2])  # (B,1,1,T)
            mask = (1.0 - mask) * -1e9
            scores = scores + mask

        # === SOFTMAX + DROPOUT + OUTPUT ===
        attn = ops.softmax(scores, axis=-1)
        attn = ops.dropout(attn, self.keep_prob, self.training)
        output = ops.matmul(attn, V)

        return output, attn