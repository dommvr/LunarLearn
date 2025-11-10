from LunarLearn.nn import Module
from LunarLearn.nn.transformer import Transformer
from LunarLearn.nn.transformer.attention import ScaledDotProductAttention
from LunarLearn.nn.layers import LayerNorm
from LunarLearn.core import Tensor, ops

class T5(Module):
    def __init__(
        self,
        vocab_size: int = 32128,
        d_model: int = 768,
        n_heads: int = 12,
        n_enc_layers: int = 12,
        n_dec_layers: int = 12,
        ff_dim: int = 3072,
        max_len: int = 512,
        keep_prob: float = 0.9,
        pretrained: bool = False
    ):
        super().__init__()
        self.transformer = Transformer(d_model=d_model,
                                       n_heads=n_heads,
                                       vocab_size=vocab_size,
                                       max_len=max_len,
                                       pos_mode="learnable",
                                       n_enc_layers=n_enc_layers,
                                       n_dec_layers=n_dec_layers,
                                       ff_dim=ff_dim,
                                       ff_activation="gelu",
                                       keep_prob=keep_prob,
                                       attention=ScaledDotProductAttention,
                                       norm=LayerNorm,
                                       norm_position="pre",
                                       use_output_head=True)
        
        if pretrained:
            self.load_state_dict(None)

    def forward(self, input_ids, decoder_input_ids=None, pad_idx=None, return_attn=False):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
        return self.transformer(src=input_ids, tgt=decoder_input_ids, pad_idx=pad_idx, return_attn=return_attn)
    
    def generate(
        self,
        input_ids,  # encoder input
        decoder_start_id=0,  # <pad> or BOS
        max_new_tokens=50,
        pad_idx=None
    ):
        self.eval()
        encoder_out = self.transformer.encoder(input_ids)
        decoder_input = Tensor([[decoder_start_id]])
        cache = None
        for _ in range(max_new_tokens):
            logits, cache = self.transformer.decoder(
                decoder_input,
                context=encoder_out,
                cache=cache,
                use_cache=True
            )
            next_token = ops.argmax(logits[:, -1:], axis=-1)
            decoder_input = ops.concatenate([decoder_input, next_token], dim=1)
            if next_token.item() == pad_idx:
                break
        return decoder_input
    

class T5Small(T5):
    def __init__(self, vocab_size, pretrained = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=512,
                         n_heads=8,
                         n_enc_layers=6,
                         n_dec_layers=6,
                         ff_dim=2048,
                         pretrained=pretrained)
        

class T5Base(T5):
    def __init__(self, vocab_size, pretrained = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=768,
                         n_heads=12,
                         n_enc_layers=12,
                         n_dec_layers=12,
                         ff_dim=3072,
                         pretrained=pretrained)
        

class T5Large(T5):
    def __init__(self, vocab_size, pretrained = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=1024,
                         n_heads=16,
                         n_enc_layers=24,
                         n_dec_layers=24,
                         ff_dim=4096,
                         pretrained=pretrained)
        

class T53B(T5):
    def __init__(self, vocab_size, pretrained = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=1024,
                         n_heads=16,
                         n_enc_layers=24,
                         n_dec_layers=24,
                         ff_dim=16384,
                         pretrained=pretrained)
        

class T511B(T5):
    def __init__(self, vocab_size, pretrained = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=1024,
                         n_heads=16,
                         n_enc_layers=24,
                         n_dec_layers=24,
                         ff_dim=65536,
                         pretrained=pretrained)