from LunarLearn.nn.layers import BaseLayer, LayerNorm, PositionalEncoding, PatchEmbedding, Dense, ClassEncoding
from LunarLearn.nn import ModuleList, SharedBlock
from LunarLearn.nn.transformer import EncoderBlock
from LunarLearn.nn.transformer.attention import ScaledDotProductAttention

class VisionTransformer(BaseLayer):
    def __init__(self,
                 patch_size=16,
                 n_classes=1000,
                 d_model=768,
                 n_heads=12,
                 pos_mode="learnable",
                 n_layers=12,
                 ff_dim=3072,
                 ff_activation="relu",
                 keep_prob=1,
                 attention=ScaledDotProductAttention,
                 norm=LayerNorm,
                 norm_position="post",
                 enc_share_weights=False,
                 use_output_head=False,
                 res_scale=1.0
                 ):
        super().__init__(trainable=True)
        self.patch_emb = PatchEmbedding(patch_size=patch_size, emb_dim=d_model)
        num_patches = self.patch_emb.num_patches
        self.cls_encoding = ClassEncoding()
        self.pos_encoding = PositionalEncoding(emb_dim=d_model, max_len=num_patches + 1, mode=pos_mode)
        if enc_share_weights:
            encoder = EncoderBlock(d_model=d_model,
                                n_heads=n_heads,
                                pos_mode=pos_mode,
                                att_keep_prob=keep_prob,
                                ff_layer1_nodes=ff_dim,
                                ff_layer2_nodes=d_model,
                                ff_activation=ff_activation,
                                ff_keep_prob=keep_prob,
                                attention=attention,
                                norm=norm,
                                norm_position=norm_position,
                                res_scale=res_scale)
            self.encoder = ModuleList([SharedBlock(encoder) for _ in range(n_layers)])
        else:
            self.encoder = ModuleList([EncoderBlock(d_model=d_model,
                                n_heads=n_heads,
                                pos_mode=pos_mode,
                                att_keep_prob=keep_prob,
                                ff_layer1_nodes=ff_dim,
                                ff_layer2_nodes=d_model,
                                ff_activation=ff_activation,
                                ff_keep_prob=keep_prob,
                                attention=attention,
                                norm=norm,
                                norm_position=norm_position,
                                res_scale=res_scale) for _ in range(n_layers)])
            
        if use_output_head:
            self.out_head = ModuleList([LayerNorm(), Dense(n_classes)])
        else:
            self.out_head = None

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.cls_encoding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        out = x[:, 0]
        if self.out_head is not None:
            out = self.out_head(out)
        return out