import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module
from LunarLearn.nn.transformer import Transformer
from LunarLearn.nn.transformer.attention import CausalScaledDotProductAttention
from LunarLearn.nn.layers import RMSNorm
from LunarLearn.core import ops

xp = backend.xp

class LLaMA(Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_heads: int = 32,
        n_kv_heads: int = 32,
        n_layers: int = 32,
        ff_dim: int = 11008,
        max_len: int = 2048,
        keep_prob: float = 1.0,
        pretrained: bool = False
    ):
        super().__init__()
        self.transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            max_len=max_len,
            pos_mode="rotary",
            n_dec_layers=n_layers,
            ff_dim=ff_dim,
            ff_activation="swish",
            keep_prob=keep_prob,
            attention=CausalScaledDotProductAttention,
            norm=RMSNorm,
            norm_position="pre",
            decoder_only=True,
            use_output_head=True
        )

        if pretrained:
            self.load_state_dict(None)

    def forward(self, input_ids, pad_idx=None, cache=None, use_cache=False):
        return self.transformer(input_ids, pad_idx=pad_idx, cache=cache, use_cache=use_cache)
    
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 50,
        pad_idx=None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        stream: bool = False
    ):
        """Autoregressive generation with KV cache."""
        self.eval()
        generated = input_ids
        cache = None

        for _ in range(max_new_tokens):
            logits, _, cache = self(generated, pad_idx=pad_idx, cache=cache, use_cache=True)
            next_logits = logits[:, -1:] / temperature

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_idx = ops.sort(next_logits, descending=True)
                cum_probs = ops.cumsum(ops.softmax(sorted_logits, axis=-1), axis=-1)
                mask = cum_probs <= top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = True
                filtered = sorted_logits.clone()
                filtered[~mask] = -xp.inf
                probs = ops.softmax(filtered, axis=-1)
                next_token = ops.multinomial(probs, num_samples=1)
                next_token = ops.gather(sorted_idx, -1, next_token)
            else:
                next_token = ops.argmax(next_logits, axis=-1)

            generated = ops.concatenate([generated, next_token], dim=1)

            if stream:
                yield generated

        return generated if not stream else None
        

class LLaMA7B(LLaMA):
    def __init__(self, vocab_size = 32000, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=4096,
                         n_heads=32,
                         n_kv_heads=32,
                         n_layers=32,
                         ff_dim=11008,
                         max_len=2048,
                         pretrained=pretrained)
        

class LLaMA13B(LLaMA):
    def __init__(self, vocab_size = 32000, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=5120,
                         n_heads=40,
                         n_kv_heads=40,
                         n_layers=40,
                         ff_dim=13824,
                         max_len=2048,
                         pretrained=pretrained)
        

class LLaMA30B(LLaMA):
    def __init__(self, vocab_size = 32000, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=6656,
                         n_heads=52,
                         n_kv_heads=52,
                         n_layers=60,
                         ff_dim=17920,
                         max_len=2048,
                         pretrained=pretrained)
        

class LLaMA65B(LLaMA):
    def __init__(self, vocab_size = 32000, pretrained: bool = False):
        super().__init__(vocab_size=vocab_size,
                         d_model=8192,
                         n_heads=64,
                         n_kv_heads=64,
                         n_layers=80,
                         ff_dim=22016,
                         max_len=2048,
                         pretrained=pretrained)