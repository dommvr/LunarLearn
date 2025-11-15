import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor, ops
import math

xp = backend.xp

# Define full NF4 codebook (16 levels for true 4-bit)
NF4_CODEBOOK = xp.array([
    -1.0, -0.707569, -0.542209, -0.416819,
    -0.310905, -0.215946, -0.127341, -0.042095,
     0.042095,  0.127341,  0.215946,  0.310905,
     0.416819,  0.542209,  0.707569,  1.0
], dtype=xp.float32)

NF4_QUANTILES   = Tensor(NF4_CODEBOOK)
NF4_DEQUANTILES = Tensor(NF4_CODEBOOK)


def pack_4bit(indices):
    """Pack flat int8 indices (0-15) into uint8 array (two per byte)."""
    indices = indices.astype(xp.uint8)  # Safe for 0-15
    if len(indices) % 2 != 0:
        indices = xp.append(indices, 0)  # Pad with 0 if odd
    packed = (indices[::2] << 4) | indices[1::2]
    return packed

def unpack_4bit(packed, original_numel: int):
    """Unpack uint8 to flat int8 indices, truncate to original_numel."""
    unpacked = xp.zeros(len(packed) * 2, dtype=xp.int8)
    unpacked[::2] = (packed >> 4).astype(xp.int8)
    unpacked[1::2] = (packed & 0x0F).astype(xp.int8)
    return unpacked[:original_numel]

def quantize_fp8(scale: Tensor) -> Tensor:
    """
    Quantize float tensor to FP8 E4M3FN (uint8 bit pattern).
    Handles NaN/inf to NaN, overflow to max (±448), rounding to nearest.
    """
    def to_fp8_e4m3fn(f: float) -> int:
        if xp.isnan(f):
            return 0x7F  # NaN
        if xp.isinf(f):
            return 0x7F  # NaN
        sign = 0x80 if f < 0 else 0
        f = abs(f)
        if f > 448:
            return sign | 0x7E  # Saturate to max normal
        if f < 2**-9:
            return sign  # Underflow to zero
        # Exponent and mantissa with rounding
        exp = math.floor(math.log2(f))
        frac = f / (2 ** exp) - 1
        # Round to nearest (ties away for simplicity; can add even)
        mant_int = int(round(frac * 8))
        if mant_int == 8:
            mant_int = 0
            exp += 1
        exp_biased = exp + 7
        if exp_biased <= 0:  # Subnormal
            shift = 1 - exp_biased
            mant_int = (8 + mant_int) >> shift  # Implicit 1, shift and round approx
            exp_biased = 0
        return sign | (exp_biased << 3) | mant_int
    v_to_fp8 = xp.vectorize(to_fp8_e4m3fn)
    q_data = v_to_fp8(scale.data).astype(xp.uint8)
    return Tensor(q_data)

def dequantize_fp8(quantized: Tensor) -> Tensor:
    """
    Dequantize FP8 E4M3FN (uint8) back to float.
    """
    def from_fp8_e4m3fn(b: int) -> float:
        b = int(b)
        if (b & 0x7F) == 0x7F:
            return xp.nan  # NaN
        sign = -1 if (b & 0x80) else 1
        exp_biased = (b >> 3) & 0x0F
        mant = b & 0x07
        if exp_biased == 0:  # Subnormal
            value = mant / 8.0 * (2 ** -6)
        else:  # Normal
            value = (1 + mant / 8.0) * (2 ** (exp_biased - 7))
        return sign * value
    v_from_fp8 = xp.vectorize(from_fp8_e4m3fn)
    dq_data = v_from_fp8(quantized.data).astype(xp.float32)  # Or fp16 if preferred
    return Tensor(dq_data)

def quantize_4bit(tensor: Tensor, block_size: int = 64) -> tuple[Tensor, Tensor, tuple]:
    """
    NF4 quantization with double quantization and 4-bit packing.
    Returns:
        packed_quantized : Tensor[uint8]   # two 4-bit values per byte
        scales_q         : Tensor[uint8]   # FP8-encoded per-block scales
        original_shape   : tuple
    """
    codebook = NF4_QUANTILES.data
    original_shape = tensor.shape
    flat = tensor.data.flatten()
    numel = flat.shape[0]
    idx_list = []
    scale_list = []

    for i in range(0, numel, block_size):
        block = flat[i:i + block_size]
        if block.size == 0:
            break  # Safety
        # NF4: map to normal distribution quantiles
        block_norm = ops.normalize_absmax(block)

        # block_norm[:, None]  (N,1)  –  codebook[None,:]  (1,16)
        diffs = ops.abs(block_norm[:, None] - codebook[None, :])
        indices = ops.argmin(diffs, dim=1)
        idx_list.extend(indices.data.tolist())

        # per-block scale (max abs)
        scale = ops.max(ops.abs(block))
        scale_list.append(scale)

    # pack indices into true 4-bit storage
    quantized_flat = xp.array(idx_list, dtype=xp.int8)
    packed_data = pack_4bit(quantized_flat)                 # uint8 ndarray
    packed_tensor = Tensor(packed_data, dtype=xp.uint8)

    # Stack *all* scales to Tensor
    scales_tensor = ops.stack(scale_list) if scale_list else Tensor(xp.array([], dtype=xp.float32))
    scales_q = quantize_fp8(scales_tensor)

    return packed_tensor, scales_q, original_shape

def dequantize_4bit(packed: Tensor, scales: Tensor, block_size: int = 64, original_shape: tuple = None) -> Tensor:
    """Dequantize packed 4-bit back to FP16."""
    if original_shape is None:
        raise ValueError("original_shape required for dequant")
    numel = xp.prod(xp.array(original_shape))

    # Unpack to flat indices (0-15)
    unpacked_flat = unpack_4bit(packed.data, numel)  # int8 array

    # Dequantize scales first (double quant)
    scales = dequantize_fp8(scales)

    # Block-wise dequant (handles partial blocks)
    dequant_flat = []
    block_idx = 0
    codebook = NF4_DEQUANTILES.data

    for i in range(scales.shape[0]):
        cur_block_sz = min(block_size, numel - block_idx)
        idx_block = unpacked_flat[block_idx : block_idx + cur_block_sz]

        scale = scales[i]
        # (or keep as Tensor and broadcast: scale * codebook[idx_block])
        deq_block = codebook[idx_block] * scale.item()
        dequant_flat.extend(deq_block)

        block_idx += cur_block_sz

    dequant_data = xp.array(dequant_flat).reshape(original_shape)
    return Tensor(dequant_data)

def fake_quantize_4bit(tensor: Tensor, block_size: int = 64) -> Tensor:
    """
    Fake quant for QAT: Quantize to 4-bit NF4, then dequantize back to float.
    Forward: Simulates quant error.
    Backward: Straight-through (grads unchanged).
    """
    if not backend.is_grad_enabled() or not tensor.requires_grad:
        return tensor  # No-op if not training

    # Quantize (your function, returns q, s, shape)
    q, s, shape = quantize_4bit(tensor, block_size=block_size)
    
    # Dequantize back to float (simulates but keeps differentiable)
    deq = dequantize_4bit(q, s, block_size=block_size, original_shape=shape)
    
    # For STE: In backward, use custom grad_fn
    out = Tensor(deq.data, requires_grad=tensor.requires_grad, dtype=tensor.dtype)
    out.is_leaf = False
    out.grad_fn = "fake_quantize_4bit"
    
    # Define backward: Straight-through (copy input grad to output)
    def _backward(grad_output: Tensor) -> Tensor:
        return grad_output  # Ignore quant; pass gradients straight through
    
    out._backward = _backward  # Assuming your Tensor has a _backward hook
    out._prev = {tensor}

    return out