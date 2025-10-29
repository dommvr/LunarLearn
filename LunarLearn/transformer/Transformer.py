import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops
from LunarLearn.models import ModuleList
from LunarLearn.transformer import Embedding, PositionalEncoding, EncoderBlock, DecoderBlock
from LunarLearn.layers import LayerNorm, Dense

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
                 norm=LayerNorm,
                 ):
        super().__init__(trainable=True)
        self.enc_embedding = Embedding(vocab_size, d_model, padding_idx, keep_prob)
        self.dec_embedding = self.enc_embedding
        self.enc_pos_encoding = PositionalEncoding(d_model, max_len, pos_enc_mode)
        self.dec_pos_encoding = PositionalEncoding(d_model, max_len, pos_enc_mode)
        self.encoder = ModuleList([EncoderBlock(d_model,
                                    n_heads,
                                    keep_prob,
                                    ff_dim,
                                    d_model,
                                    ff_activation,
                                    keep_prob,
                                    norm) for _ in range(n_enc_layers)])
        self.decoder = ModuleList([DecoderBlock(d_model,
                                    n_heads,
                                    keep_prob,
                                    keep_prob,
                                    ff_dim,
                                    d_model,
                                    ff_activation,
                                    keep_prob,
                                    norm)for _ in range(n_dec_layers)])
        self.linear = Dense(vocab_size, transpose_weight=True)
        self.linear.W = self.enc_embedding.W
        
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        mask = ops.tril(xp.ones((tgt.shape[1], tgt.shape[1]), dtype=DTYPE))
        x = self.enc_embedding(src)

        if self.linear.W is None and self.enc_embedding.W is not None:
            self.linear.W = self.enc_embedding.W

        x = self.enc_pos_encoding(x)
        enc_out = self.encoder(x)

        x = self.dec_embedding(tgt)
        x = self.dec_pos_encoding(x)
        dec_out = self.decoder(x, mask=mask, context=enc_out)

        out = self.linear(dec_out)
        out = ops.softmax(out, axis=-1)
        return out
        
