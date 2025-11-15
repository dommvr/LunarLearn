import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module
from LunarLearn.core import Tensor
from LunarLearn.quantization.utils import quantize_4bit, dequantize_4bit

class QuantizedModel(Module):
    def __init__(self, model, block_size: int = 64, quantize: bool = True):
        super.__init__()
        self.model = model
        self.block_size = block_size
        self.quantize = quantize
        self._quant_cache = dict[str, tuple[Tensor, Tensor]] = {}

    def _ensure_quantised(self):
        """Quantise every parameter that is not yet in the cache."""
        if not self.quantize:
            return

        for name, param in self.model.named_parameters():
            if name in self._quant_cache:
                continue

            # ``param.master`` is the FP32 tensor we quantise from
            q, s = quantize_4bit(param.master, block_size=self.block_size)
            self._quant_cache[name] = (q, s)

    def forward(self, *args, **kwargs):
        if not self.quantize:
            return self.model(*args, **kwargs)
        
        self._ensure_quantised()

        with backend.no_grad():
            for name, param in self.model.name_parameters():
                q, s, shape = self._quant_cache[name]
                param.master = dequantize_4bit(q, s, block_size=self.block_size, original_shape=shape)
        
        return self.model(*args, **kwargs)