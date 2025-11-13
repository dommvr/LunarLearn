import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module
from LunarLearn.quantization import quantize_4bit, dequantize_4bit

class QuantizedModel(Module):
    def __init__(self, model, block_size: int = 64, quantize: bool = True):
        super.__init__()
        self.model = model
        self.block_size = block_size
        self.quantize = quantize
        self.quantized_states = {}

    def forward(self, *args, **kwargs):
        if self.quantize:
            for name, param in self.model.named_parameters():
                if name not in self.quantized_states:
                    self.quantized_states[name] = quantize_4bit(param.master, block_size=self.block_size)
            with backend.no_grad():
                for name, param in self.model_parameters():
                    q, s = self.quantized_states[name]
                    param.master = dequantize_4bit(q, s, block_size=self.block_size)
            out = self.model(*args, **kwargs)
            for name, param in self.model.named_parameters():
                q, s = self.quantized_states[name]
                q = quantize_4bit(param.master)[0]
            return out 
        else:
            return self.model(*args, **kwargs)