from LunarLearn.nn import Module
from LunarLearn.nn.transformer import Transformer
from LunarLearn.nn.transformer.attention import CausalScaledDotProductAttention
from LunarLearn.nn.layers import LayerNorm
from LunarLearn.core import Tensor

class GPT2(Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_len: int = 1024,
        keep_prob: float = 0.9,
        pretrained: bool = False
        ):
        super().__init__()
        self.transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            vocab_size=vocab_size,
            max_len=max_len,
            pos_mode="learnable",
            n_dec_layers=n_layers,
            ff_dim=d_model * 4,
            ff_activation="gelu",
            keep_prob=keep_prob,
            attention=CausalScaledDotProductAttention,
            norm=LayerNorm,
            decoder_only=True,
            use_output_head=True
        )

        if pretrained:
            self.load_state_dict(None)
        
    def forward(self, input_ids: Tensor, pad_idx=None, return_attn=False) -> Tensor:
        return self.transformer(input_ids, pad_idx, return_attn)
    

class GPT2Small(GPT2):
    def __init__(self, vocab_size, pretrained=False):
        super().__init__(vocab_size=vocab_size,
                         d_model=768,
                         n_heads=12,
                         n_layers=12,
                         pretrained=pretrained)
        

class GPT2Medium(GPT2):
    def __init__(self, vocab_size, pretrained=False):
        super().__init__(vocab_size=vocab_size,
                         d_model=1024,
                         n_heads=16,
                         n_layers=24,
                         pretrained=pretrained)
        

class GPT2Large(GPT2):
    def __init__(self, vocab_size, pretrained=False):
        super().__init__(vocab_size=vocab_size,
                         d_model=1280,
                         n_heads=20,
                         n_layers=36,
                         pretrained=pretrained)