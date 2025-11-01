import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops
from LunarLearn.models import ModuleList, Sequential
from LunarLearn.transformer import Embedding, PositionalEncoding, EncoderBlock, DecoderBlock
from LunarLearn.layers import LayerNorm, Dense
from LunarLearn.transformer.attention import ScaledDotProductAttention
from LunarLearn.transformer.utils.masks import make_pad_mask, make_causal_mask, merge_masks

xp = backend.xp
DTYPE = backend.DTYPE

class Transformer(BaseLayer):
    def __init__(self,
                 d_model=512,
                 n_heads=8,
                 vocab_size=32000,
                 padding_idx=None,
                 max_len=512,
                 pos_mode="learnable",
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
        self.enc_pos_encoding = PositionalEncoding(d_model, max_len, pos_mode)
        self.dec_pos_encoding = PositionalEncoding(d_model, max_len, pos_mode)
        if enc_share_weights and not decoder_only:
            shared_encoder = EncoderBlock(d_model,
                                        n_heads,
                                        pos_mode,
                                        keep_prob,
                                        ff_dim,
                                        d_model,
                                        ff_activation,
                                        keep_prob,
                                        attention,
                                        norm,
                                        norm_position,
                                        res_scale)
            self.encoderblock = [lambda x, mask=None, return_attn=False: shared_encoder(x, mask, return_attn)] * n_enc_layers
        elif not enc_share_weights and not decoder_only:
            self.encoderblock = ModuleList([EncoderBlock(d_model,
                                        n_heads,
                                        pos_mode,
                                        keep_prob,
                                        ff_dim,
                                        d_model,
                                        ff_activation,
                                        keep_prob,
                                        attention,
                                        norm,
                                        norm_position,
                                        res_scale) for _ in range(n_enc_layers)])
        self.decoderblock = ModuleList([DecoderBlock(d_model,
                                    n_heads,
                                    pos_mode,
                                    keep_prob,
                                    keep_prob,
                                    ff_dim,
                                    d_model,
                                    ff_activation,
                                    keep_prob,
                                    attention,
                                    norm,
                                    norm_position,
                                    res_scale,
                                    cross_attn=not(decoder_only)) for _ in range(n_dec_layers)])
        self.linear = Dense(vocab_size, transpose_weight=True)
        self.linear.W = self.enc_embedding.W

        if use_output_head:
            self.out_head = Sequential(norm(), Dense(d_model, activation=ff_activation, keep_prob=keep_prob))

        self.decoder_only = decoder_only

    def encoder(self, src: Tensor, mask=None, return_attn=False) -> Tensor:
        attn_list = []
        x = self.enc_embedding(src)
        if self.linear.W is None and self.enc_embedding.W is not None:
            self.linear.W = self.enc_embedding.W
        x = self.enc_pos_encoding(x)
        for layer in self.encoderblock:
            x, attn = layer(x, mask=mask, return_attn=True)
            if return_attn:
                attn_list.append(attn)
        return x, attn_list if return_attn else x

    def decoder(self, tgt: Tensor, mask=None, context=None, return_attn=False):
        attn_list = []
        x = self.dec_embedding(tgt)
        x = self.dec_pos_encoding(x)
        for layer in self.decoderblock:
            x, attn = layer(x, mask=mask, context=context, return_attn=True)
            if return_attn:
                attn_list.append(attn)
        return x, attn_list if return_attn else x

    def forward(self, src: Tensor, tgt: Tensor, pad_idx=None, return_attn=False) -> Tensor:
        pad_mask = make_pad_mask(src, pad_idx)
        causal_mask = make_causal_mask(tgt.shape[1])
        mask = merge_masks(pad_mask, causal_mask)

        if not self.decoder_only:
            enc_out, enc_attn = self.encoder(src, return_attn=return_attn)
        out, dec_attn = self.decoder(tgt, mask=mask, context=enc_out if not self.decoder_only else None, return_attn=return_attn)

        if self.out_head is not None:
            out = self.out_head(out)

        out = self.linear(out)
        out = ops.softmax(out, axis=-1)
        return (out, enc_attn, dec_attn) if return_attn else (out, None, None)