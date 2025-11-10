from LunarLearn.nn import Module
from LunarLearn.nn.transformer import Transformer
from LunarLearn.nn.transformer.attention import CausalScaledDotProductAttention
from LunarLearn.nn.layers import LayerNorm
from LunarLearn.core import Tensor, ops

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
    
    def generate(
        self,
        input_ids,
        max_new_tokens=50,
        pad_idx=None,
        temperature=1.0,
        top_p=0.9,
        stream=False
    ):
        """
        Autoregressive generation with optional streaming.

        Args:
            input_ids (Tensor): (B, L) starting tokens
            max_new_tokens (int): Max tokens to generate
            pad_idx (int, optional): Padding token ID
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling threshold
            stream (bool): If True, yield partial sequences

        Yields:
            Tensor: Generated sequence so far (if stream=True)

        Returns:
            Tensor: Final generated sequence (if stream=False)
        """
        self.eval()
        generated = input_ids
        cache = None
        for _ in range(max_new_tokens):
            logits, cache = self(generated, pad_idx=pad_idx, cache=cache, use_cache=True)
            next_logits = logits[:, -1:] / temperature
            if top_p < 1.0:
                # top-p sampling
                probs = ops.softmax(next_logits, axis=-1)
                sorted_probs, sorted_idx = ops.sort(probs, descending=True)
                cum_probs = ops.cumsum(sorted_probs, axis=-1)
                mask = cum_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs /= sorted_probs.sum(axis=-1, keepdims=True)
                next_token = ops.multinomial(sorted_probs, 1)
                next_token = ops.gather(sorted_idx, -1, next_token)
            else:
                next_token = ops.argmax(next_logits, axis=-1)
            generated = ops.concatenate([generated, next_token], dim=1)

            if stream:
                yield generated

        return generated if not stream else None
    

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
        

class GPT2XL(GPT2):
    def __init__(self, vocab_size, pretrained=False):
        super().__init__(vocab_size=vocab_size,
                         d_model=1600,
                         n_heads=25,
                         n_layers=48,
                         pretrained=pretrained)