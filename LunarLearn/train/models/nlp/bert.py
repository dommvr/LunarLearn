from LunarLearn.nn import Module, ModuleList
from LunarLearn.core import Tensor
from LunarLearn.nn.transformer import Transformer
from LunarLearn.nn.transformer.attention import ScaledDotProductAttention
from LunarLearn.nn.layers import LayerNorm, Dense, Activation

class BERT(Module):
    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_len: int = 512,
        keep_prob: float = 0.9,
        use_mlm_head: bool = True,
        use_nsp_head: bool = True,
        pretrained: bool = False
    ):
        super().__init__()
        self.transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            vocab_size=vocab_size,
            max_len=max_len,
            pos_mode="learnable",
            n_enc_layers=n_layers,
            ff_dim=d_model * 4,
            ff_activation="gelu",
            keep_prob=keep_prob,
            attention=ScaledDotProductAttention,
            norm=LayerNorm,
            norm_position="pre",
            encoder_only=True,
            use_output_head=False
        )
        self.use_mlm_head = use_mlm_head
        self.use_nsp_head = use_nsp_head

        if use_mlm_head:
            self.mlm_head = ModuleList([
                Dense(d_model),
                Activation("gelu"),
                LayerNorm(),
                Dense(vocab_size, transpose_weight=True)
            ])

        if use_nsp_head:
            self.nsp_head = ModuleList([
                Dense(d_model),
                Activation("gelu"),
                Dense(2)
            ])

        if pretrained:
            self.load_state_dict(None)

    def forward(self, input_ids: Tensor, pad_idx=None, return_hidden=False, return_attn=False):
        hidden, attn, cache = self.transformer(input_ids, pad_idx=pad_idx, return_attn=return_attn)

        if return_hidden:
            return hidden
        
        outputs = {}
        if self.use_mlm_head:
            if self.mlm_head[-1].W is None and self.transformer.dec_embedding.W is not None:
                self.mlm_head[-1].W = self.transformer.dec_embedding.W 
            outputs["mlm_logits"] = self.mlm_head(hidden)
        if self.use_nsp_head:
            cls_hidden = hidden[:, 0]
            outputs["nsp_logits"] = self.nsp_head(cls_hidden)

        return (outputs, attn, cache) if outputs else (hidden, attn, cache)
    

class BERTTiny(BERT):
    def __init__(self, vocab_size, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=128,
                         n_heads=2,
                         n_layers=2,
                         pretrained=pretrained)
        

class BERTMini(BERT):
    def __init__(self, vocab_size, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=256,
                         n_heads=4,
                         n_layers=4,
                         pretrained=pretrained)


class BERTSmall(BERT):
    def __init__(self, vocab_size, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=512,
                         n_heads=8,
                         n_layers=4,
                         pretrained=pretrained)
        

class BERTMedium(BERT):
    def __init__(self, vocab_size, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=512,
                         n_heads=8,
                         n_layers=8,
                         pretrained=pretrained)


class BERTLarge(BERT):
    def __init__(self, vocab_size, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=1024,
                         n_heads=16,
                         n_layers=24,
                         pretrained=pretrained)