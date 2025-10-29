from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.layers import LayerNorm
from LunarLearn.transformer import MultiHeadAttention, FeedForward
from LunarLearn.tensor import Tensor

class EncoderBlock(BaseLayer):
    def __init__(self,
                 d_model,
                 n_heads,
                 att_keep_prob=1.0,
                 ff_layer1_nodes=2048,
                 ff_layer2_nodes=512,
                 ff_activation="relu",
                 ff_keep_prob=1.0,
                 norm=LayerNorm,
                 norm_position="post"): #pre
        super().__init__(trainable=True)
        self.mhattention = MultiHeadAttention(d_model, n_heads, att_keep_prob)
        self.norm1 = norm(axis=-1)
        self.norm2 = norm(axis=-1)

        self.feedforward = FeedForward(ff_layer1_nodes, ff_layer2_nodes, ff_activation, ff_keep_prob)

        self.norm_position=norm_position
    
    def forward(self, x: Tensor, mask=None) -> Tensor:
        # Multi-head self-attention
        attn_out = self.mhattention(x, mask=mask)
        if self.norm_position == "pre":
            attn_out = self.norm1(attn_out)
            x += attn_out
        else:
            x = self.norm1(x + attn_out)

        # Feed-forward network
        ff_out = self.feedforward(x)
        if self.norm_position == "pre":
            ff_out = self.norm2(ff_out)
            out = x + ff_out
        else:
            out = self.norm2(x + ff_out)
        return out