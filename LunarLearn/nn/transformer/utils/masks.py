import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

def make_pad_mask(seq: Tensor, pad_idx=0):
    # seq: (B, L)
    mask = (seq != pad_idx).astype(DTYPE)
    return mask[:, xp.newaxis, xp.newaxis, :]  # (B,1,1,L)

def make_causal_mask(L):
    return xp.tril(xp.ones((L, L), dtype=DTYPE))[xp.newaxis, xp.newaxis, :, :]

def merge_masks(pad_mask=None, causal_mask=None):
    if pad_mask is None: 
        return causal_mask
    if causal_mask is None:
        return pad_mask
    return pad_mask * causal_mask

def _window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return x

def get_shifted_window_mask(H: int, W: int, window_size: int, shift_size: int) -> Tensor:
    """
    Generate attention mask for shifted windows in Swin Transformer.
    
    Args:
        H (int): Height of feature map.
        W (int): Width of feature map.
        window_size (int): Local window size.
        shift_size (int): Shift size for SW-MSA.
    
    Returns:
        Tensor: Attention mask of shape (num_windows, N, N) where N = window_size ** 2.
    """
    # Create image mask with region IDs
    img_mask = ops.zeros((1, H, W, 1))
    
    # Define slices for regions created by shift
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    
    # Partition into windows (same as input partitioning)
    mask_windows = _window_partition(img_mask, window_size)  # (num_windows, ws, ws, 1)
    mask_windows = mask_windows.reshape(-1, window_size * window_size)  # (num_windows, N)
    
    # Compute mask: -100 where different regions, 0 where same
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (num_windows, N, N)
    attn_mask = ops.where(attn_mask != 0, ops.full_like(attn_mask, -100.0), ops.full_like(attn_mask, 0.0))
    
    return attn_mask