import LunarLearn.backend as backend
from LunarLearn.tensor import Tensor

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