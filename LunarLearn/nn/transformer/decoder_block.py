from LunarLearn.nn.layers import BaseLayer, LayerNorm
from LunarLearn.nn.transformer import MultiHeadAttention, FeedForward
from LunarLearn.nn.transformer.attention import ScaledDotProductAttention
from LunarLearn.core import Tensor

class DecoderBlock(BaseLayer):
    def __init__(self,
                 d_model,
                 n_heads,
                 pos_mode=None,
                 att1_keep_prob=1.0,
                 att2_keep_prob=1.0,
                 ff_layer1_nodes=2048,
                 ff_layer2_nodes=512,
                 ff_activation="relu",
                 ff_keep_prob=1.0,
                 attention=ScaledDotProductAttention,
                 norm=LayerNorm,
                 norm_position="post",
                 res_scale=1.0,
                 use_cross_attn=True): #pre
        super().__init__(trainable=True)
        self.self_attn = MultiHeadAttention(d_model, n_heads, attention, pos_mode, att1_keep_prob)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, attention, pos_mode, att2_keep_prob)

        self.norm1 = norm(axis=-1)
        self.norm2 = norm(axis=-1)
        self.norm3 = norm(axis=-1)

        self.feedforward = FeedForward(ff_layer1_nodes, ff_layer2_nodes, ff_activation, ff_keep_prob)

        self.norm_position = norm_position
        self.res_scale = res_scale
        self.use_cross_attn = use_cross_attn

    def forward(self, x: Tensor, mask=None, context=None, return_attn=False) -> Tensor:
        if self.norm_position == "pre":
            attn1_out, attn1 = self.self_attn(self.norm1(x), mask=mask)
            x = x + attn1_out * self.res_scale

            if self.cross_attn:
                attn2_out, attn2 = self.cross_attn(self.norm2(x), context=context)
                x = x + attn2_out * self.res_scale

            ff_out = self.feedforward(self.norm3(x))
            out = x + ff_out * self.res_scale
        else:
            attn1_out, attn1 = self.self_attn(x, mask=mask)
            x = self.norm1(x + attn1_out * self.res_scale)

            if self.cross_attn:
                attn2_out, attn2 = self.cross_attn(x, context=context)
                x = self.norm2(x + attn2_out * self.res_scale)

            ff_out = self.feedforward(x)
            out = self.norm3(x + ff_out * self.res_scale)
        return (out, {"self": attn1, "cross": attn2 if self.use_cross_attn else None}) if return_attn else (out, None)