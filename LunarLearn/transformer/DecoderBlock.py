from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.transformer import MultiHeadAttention, FeedForward
from LunarLearn.layers import LayerNorm
from LunarLearn.tensor import Tensor

class DecoderBlock(BaseLayer):
    def __init__(self,
                 d_model,
                 n_heads,
                 att1_keep_prob=1.0,
                 att2_keep_prob=1.0,
                 ff_layer1_nodes=2048,
                 ff_layer2_nodes=512,
                 ff_activation="relu",
                 ff_keep_prob=1.0,
                 norm=LayerNorm):
        super().__init__(trainable=True)
        self.self_attn = MultiHeadAttention(d_model, n_heads, att1_keep_prob)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, att2_keep_prob)

        self.norm1 = norm(axis=-1)
        self.norm2 = norm(axis=-1)
        self.norm3 = norm(axis=-1)

        self.feedforward = FeedForward(ff_layer1_nodes, ff_layer2_nodes, ff_activation, ff_keep_prob)

    def forward(self, x: Tensor, mask=None, context=None) -> Tensor:
        attn1_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + attn1_out)

        attn2_out = self.cross_attn(x, context=context)
        x = self.norm2(x + attn2_out)

        ff_out = self.feedforward(x)
        out = self.norm3(x + ff_out)
        return out