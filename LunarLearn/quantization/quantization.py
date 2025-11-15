from contextlib import contextmanager
from LunarLearn.core import Tensor
from LunarLearn.quantization.utils import fake_quantize_4bit

def quantization_autocast(bits: int = 4, block_size: int = 64, enabled: bool = True):
    """
    Context for QAT: Applies fake quant to activations/weights in forward.
    Usage: with quantization_autocast(): model(inputs)
    """
    if not enabled:
        yield
        return
    
    # Temporarily add hooks to tensors or layers (your library's way)
    # Example: Assume you have global activation hooks like in your sort/searchsorted
    def qat_hook(tensor: Tensor) -> Tensor:
        return fake_quantize_4bit(tensor, block_size=block_size)
    
    # Attach hook to Tensor creations or layer outputs (customize to your lib)
    Tensor._activation_hooks.append(qat_hook)  # If you have this system
    
    # For weights: Iterate model params and fake quant master
    for param in model.parameters():  # Assuming global model or pass it
        param.master = fake_quantize_4bit(param.master, block_size=block_size)
    
    try:
        yield
    finally:
        # Clean up hooks
        Tensor._activation_hooks.remove(qat_hook)
        # No need to revert weights; they'll update via STE