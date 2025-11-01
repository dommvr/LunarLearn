import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops
from LunarLearn.models import ModuleList, Sequential
from LunarLearn.transformer import Embedding, PositionalEncoding, EncoderBlock, DecoderBlock
from LunarLearn.layers import LayerNorm, Dense, LambdaLayer
from LunarLearn.transformer.attention import ScaledDotProductAttention

xp = backend.xp
DTYPE = backend.DTYPE

class Transformer(BaseLayer):
    def __init__(self,
                 d_model=512,
                 n_heads=8,
                 vocab_size=32000,
                 padding_idx=None,
                 max_len=512,
                 pos_enc_mode="learnable",
                 n_enc_layers=6,
                 n_dec_layers=6,
                 ff_dim=2048,
                 ff_activation="relu",
                 keep_prob=1.0,
                 attention=ScaledDotProductAttention,
                 norm=LayerNorm,
                 norm_position="post",
                 enc_share_weights=False,
                 use_output_head=False,
                 decoder_only=False,
                 res_scale=1.0
                 ):
        super().__init__(trainable=True)
        self.enc_embedding = Embedding(vocab_size, d_model, padding_idx, keep_prob)
        self.dec_embedding = self.enc_embedding
        self.enc_pos_encoding = PositionalEncoding(d_model, max_len, pos_enc_mode)
        self.dec_pos_encoding = PositionalEncoding(d_model, max_len, pos_enc_mode)
        if enc_share_weights and not decoder_only:
            encoder = EncoderBlock(d_model,
                                        n_heads,
                                        keep_prob,
                                        ff_dim,
                                        d_model,
                                        ff_activation,
                                        keep_prob,
                                        attention,
                                        norm,
                                        norm_position)
            self.encoderblock = LambdaLayer(lambda x: encoder(x) for _ in range(n_enc_layers))
        elif not enc_share_weights and not decoder_only:
            self.encoderblock = ModuleList([EncoderBlock(d_model,
                                        n_heads,
                                        keep_prob,
                                        ff_dim,
                                        d_model,
                                        ff_activation,
                                        keep_prob,
                                        attention,
                                        norm,
                                        norm_position) for _ in range(n_enc_layers)])
        self.decoderblock = ModuleList([DecoderBlock(d_model,
                                    n_heads,
                                    keep_prob,
                                    keep_prob,
                                    ff_dim,
                                    d_model,
                                    ff_activation,
                                    keep_prob,
                                    attention,
                                    norm,
                                    norm_position,
                                    cross_attn=not(decoder_only))for _ in range(n_dec_layers)])
        self.linear = Dense(vocab_size, transpose_weight=True)
        self.linear.W = self.enc_embedding.W

        if use_output_head:
            self.out_head = Sequential(norm(), Dense(d_model, activation=ff_activation, keep_prob=keep_prob))

        self.decoder_only = decoder_only

    def encoder(self, src: Tensor) -> Tensor:
        x = self.enc_embedding(src)
        if self.linear.W is None and self.enc_embedding.W is not None:
            self.linear.W = self.enc_embedding.W
        x = self.enc_pos_encoding(x)
        out = self.encoderblock(x)
        return out

    def decoder(self, tgt: Tensor, mask=None, context=None):
        x = self.dec_embedding(tgt)
        x = self.dec_pos_encoding(x)
        out = self.decoderblock(x, mask=mask, context=context)
        return out

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        mask = ops.tril(xp.ones((tgt.shape[1], tgt.shape[1]), dtype=DTYPE))
        if not self.decoder_only:
            enc_out = self.encoder(src)
        out = self.decoder(tgt, mask=mask, context=enc_out if not self.decoder_only else None)

        if self.out_head is not None:
            out = self.out_head(out)

        out = self.linear(out)
        out = ops.softmax(out, axis=-1)
        return out
        
