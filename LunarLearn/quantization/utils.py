import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor, ops

xp = backend.xp

# NF4 constants (precomputed)
NF4_QUANTILES = Tensor([-1.0, -0.5, 0.0, 0.5, 1.0])  # 4-bit normal
NF4_DEQUANTILES = Tensor([-1.0, -0.3734, 0.0, 0.3734, 1.0])


def quantize_4bit(tensor: Tensor, block_size: int = 64) -> tuple[Tensor, Tensor]:
    """
    NF4 quantization with double quantization.
    Returns: quantized (int4), scales (fp8)
    """
    # Block-wise quantization
    flat = tensor.data.flatten()
    blocks = flat.reshape(-1, block_size)
    quantized = []
    scales = []

    for block in blocks:
        # NF4: map to normal distribution quantiles
        block_norm = ops.normalize(block)
        indices = ops.searchsorted(NF4_QUANTILES, block_norm)
        quantized.append(indices.astype(xp.int8))

        # Double quant: quantize scales to 8-bit
        scale = ops.max(ops.abs(block)) / (2**3)  # FP8
        scales.append(ops.quantize_fp8(scale))

    quantized = ops.stack(quantized).reshape(tensor.shape)
    scales = ops.stack(scales)

    return Tensor(quantized), Tensor(scales)

def dequantize_4bit(quantized: Tensor, scales: Tensor, block_size: int = 64) -> Tensor:
    """Dequantize back to FP16."""
    # Dequantize scales first (double quant)
    scales = ops.dequantize_fp8(scales)

    # Block-wise dequant
    flat = quantized.data.flatten()
    blocks = flat.reshape(-1, block_size)
    dequant = []

    for i, block in enumerate(blocks):
        scale = scales[i // block_size]
        dequant_block = NF4_DEQUANTILES[block] * scale
        dequant.append(dequant_block)

    return Tensor(ops.stack(dequant).reshape(quantized.shape))