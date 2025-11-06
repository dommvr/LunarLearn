from LunarLearn.nn.layers import BaseLayer, LayerNorm
from LunarLearn.nn.transformer import MultiHeadAttention, FeedForward
from LunarLearn.nn.transformer.attention import ScaledDotProductAttention
from LunarLearn.core import Tensor

class EncoderBlock(BaseLayer):
    def __init__(self,
                 d_model,
                 n_heads,
                 pos_mode=None,
                 att_keep_prob=1.0,
                 ff_layer1_nodes=2048,
                 ff_layer2_nodes=512,
                 ff_activation="relu",
                 ff_keep_prob=1.0,
                 attention=ScaledDotProductAttention,
                 norm=LayerNorm,
                 norm_position="post",
                 res_scale=1.0):
        super().__init__(trainable=True)
        self.mhattention = MultiHeadAttention(d_model, n_heads, attention, pos_mode, att_keep_prob)
        self.norm1 = norm(axis=-1)
        self.norm2 = norm(axis=-1)

        self.feedforward = FeedForward(ff_layer1_nodes, ff_layer2_nodes, ff_activation, ff_keep_prob)

        self.norm_position=norm_position
        self.res_scale = res_scale
    
    def forward(self, x: Tensor, mask=None, return_attn=False) -> Tensor:
        if self.norm_position == "pre":
            attn_out, attn = self.mhattention(self.norm1(x), mask=mask)
            x = x + attn_out * self.res_scale

            ff_out = self.feedforward(self.norm2(x))
            out = x + ff_out * self.res_scale
        else:
            # Multi-head self-attention
            attn_out, attn = self.mhattention(x, mask=mask)
            x = self.norm1(x + attn_out * self.res_scale)

            # Feed-forward network
            ff_out = self.feedforward(x)
            out = self.norm2(x + ff_out * self.res_scale)
        return (out, attn) if return_attn else (out, None)