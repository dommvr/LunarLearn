import numpy as np
import cupy as cp

import LunarLearn.core.backend.backend as backend
from LunarLearn.data.dataloader import DataLoader, SubsetDataset
from LunarLearn.core import Tensor

xp = backend.xp


def _is_np_array(x):
    return isinstance(x, np.ndarray)


def _is_np_scalar(x):
    return isinstance(x, (np.generic,))


def _try_stack_np(batch):
    """
    Try to np.stack a list of arrays. If ragged or incompatible, return list(batch).
    """
    try:
        return np.stack(batch, axis=0)
    except Exception:
        return list(batch)


def _is_xp_array(x):
    return isinstance(x, cp.ndarray) or isinstance(x, np.ndarray)


def _try_stack(arrs):
    # arrs: list of xp arrays
    try:
        return xp.stack(arrs, axis=0)
    except Exception:
        return arrs  # variable shapes -> keep list


def _try_stack_any(batch):
    """
    Stacks xp arrays or Tensors if possible; otherwise returns list.
    """
    first = batch[0]
    if isinstance(first, Tensor):
        try:
            data = [b.data if isinstance(b, Tensor) else b for b in batch]
            return Tensor(xp.stack(data, axis=0), requires_grad=False)
        except Exception:
            return list(batch)
    if _is_xp_array(first):
        return _try_stack(batch)
    return list(batch)


def _to_tensor_tree(obj):
    """
    Recursively convert xp arrays to Tensor.
    Leaves strings alone.
    Keeps lists for variable-length stuff.
    """
    if isinstance(obj, Tensor):
        return obj
    if isinstance(obj, dict):
        return {k: _to_tensor_tree(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_tensor_tree(v) for v in obj)
    if isinstance(obj, str):
        return obj
    if _is_xp_array(obj):
        return Tensor(obj, requires_grad=False)
    return obj


def _read_text_file(path, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def _tokenize_text(text, tokenizer):
    """
    tokenizer can return:
      - list[int]
      - xp array
      - numpy array
    """
    out = tokenizer.encode(text)
    return np.asarray(out, dtype=np.int64)


def _pad_1d(ids, max_len, pad_id=0, truncate=True):
    """
    ids: xp int64 1D
    returns (padded_ids, attn_mask, length)
    """
    if not _is_xp_array(ids):
        ids = np.asarray(ids, dtype=np.int64)
    ids = ids.astype(np.int64)

    L = int(ids.shape[0])
    if truncate and L > max_len:
        ids = ids[:max_len]
        L = max_len

    out = np.full((max_len,), int(pad_id), dtype=np.int64)
    out[:L] = ids
    mask = np.zeros((max_len,), dtype=np.int64)
    mask[:L] = 1
    return out, mask, np.asarray(L, dtype=np.int64)


def random_split(dataset, lengths, random_state=None):
    """
    Split a dataset into multiple SubsetDatasets with given lengths.
    Example: train,val = random_split(ds, [5000, 1000], random_state=123)

    Note: uses xp RNG if available; otherwise deterministic python fallback.
    """
    n = len(dataset)
    if sum(lengths) != n:
        raise ValueError("sum(lengths) must equal len(dataset)")

    # permutation
    if random_state is None:
        perm = xp.random.permutation(n)
    else:
        rs = xp.random.RandomState(int(random_state))
        perm = rs.permutation(n)

    perm = [int(i) for i in perm.tolist()]
    out = []
    start = 0
    for L in lengths:
        out.append(SubsetDataset(dataset, perm[start:start + int(L)]))
        start += int(L)
    return out


def compute_channel_mean_std(
    dataset,
    channel_dim=0,
    eps=1e-8,
    max_samples=None,
    get_x=None,
):
    """
    Compute per-channel mean/std over a map-style dataset by iterating dataset[i].

    Assumptions:
      - Each sample x is an image-like array with shape (C,H,W) or (H,W,C) etc.
      - You choose channel_dim relative to the SAMPLE (not batched) array.
        Examples:
          channel-first sample: (C,H,W) -> channel_dim=0
          channel-last  sample: (H,W,C) -> channel_dim=-1

    Args:
      dataset: supports __len__ and __getitem__.
      channel_dim: channel axis in the sample x.
      eps: numerical stability for std.
      max_samples: optional cap for faster estimation (uses first N samples).
      get_x: optional function(sample)->x to extract x from arbitrary structures.
             If None, tries: dict["x"], tuple/list[0], else sample itself.

    Returns:
      mean: xp.ndarray shape (C,)
      std:  xp.ndarray shape (C,)
    """
    m = len(dataset)
    if max_samples is not None:
        m = min(m, int(max_samples))

    sum_c = None
    sumsq_c = None
    n_total = 0  # total number of pixels per channel accumulated

    def _default_get_x(sample):
        if isinstance(sample, dict):
            if "x" in sample:
                return sample["x"]
            # fall back to first value if you insist on chaos
            return next(iter(sample.values()))
        if isinstance(sample, (tuple, list)):
            return sample[0]
        return sample

    extractor = get_x if get_x is not None else _default_get_x

    for i in range(m):
        sample = dataset[i]
        x = extractor(sample)

        # Move to backend array
        x = xp.asarray(x)

        # Cast to float for meaningful statistics (uint8 will betray you)
        x = x.astype(xp.float32, copy=False)

        if x.ndim < 2:
            raise ValueError(f"Expected image-like sample with ndim>=2, got shape {x.shape} at index {i}")

        cd = channel_dim if channel_dim >= 0 else (x.ndim + channel_dim)
        if cd < 0 or cd >= x.ndim:
            raise ValueError(f"channel_dim={channel_dim} invalid for sample shape {x.shape} at index {i}")

        C = x.shape[cd]

        # Sum over all axes except channel axis
        axes = tuple(ax for ax in range(x.ndim) if ax != cd)

        s = x.sum(axis=axes)          # (C,)
        ss = (x * x).sum(axis=axes)   # (C,)
        count = 1
        for ax in axes:
            count *= x.shape[ax]      # pixels per channel for this sample

        if sum_c is None:
            sum_c = xp.zeros((C,), dtype=xp.float32)
            sumsq_c = xp.zeros((C,), dtype=xp.float32)

        # Safety: ensure consistent channel count across samples
        if s.shape[0] != sum_c.shape[0]:
            raise ValueError(
                f"Inconsistent channel count: expected {sum_c.shape[0]}, got {s.shape[0]} "
                f"for sample shape {x.shape} at index {i}"
            )

        sum_c += s
        sumsq_c += ss
        n_total += count

    mean = sum_c / max(n_total, 1)
    var = sumsq_c / max(n_total, 1) - mean * mean
    var = xp.maximum(var, 0.0)
    std = xp.sqrt(var + eps)

    return mean, std


def to_backend(batch, *, dtype=None, wrap_tensors=False):
    """
    Recursively move a collated CPU batch (NumPy / Python structures) to the current backend xp.

    Rules:
      - np.ndarray -> xp.asarray (one big transfer for stacked arrays)
      - python scalars / np scalars -> xp.asarray (vector/scalar)
      - dict/list/tuple -> recurse
      - str -> keep as-is
      - None -> keep as-is
      - Tensor -> leave as-is by default
      - xp arrays -> leave as-is
    Args:
      dtype: if not None, casts numeric arrays to this dtype on backend.
             (Use carefully: token ids should stay int64.)
      wrap_tensors: if True, wrap xp arrays into Tensor(requires_grad=False) recursively.
    """
    # None
    if batch is None:
        return None

    # Strings
    if isinstance(batch, str):
        return batch

    # Tensor: keep as-is (assume user knows what they did)
    if isinstance(batch, Tensor):
        return batch

    # xp arrays: already on backend
    if _is_xp_array(batch):
        out = batch
        if dtype is not None:
            out = xp.asarray(out, dtype=dtype)
        if wrap_tensors:
            out = Tensor(out, requires_grad=False)
        return out

    # NumPy arrays -> backend
    if _is_np_array(batch):
        out = xp.asarray(batch, dtype=(dtype if dtype is not None else None))
        if wrap_tensors:
            out = Tensor(out, requires_grad=False)
        return out

    # dict
    if isinstance(batch, dict):
        return {k: to_backend(v, dtype=dtype, wrap_tensors=wrap_tensors) for k, v in batch.items()}

    # list/tuple
    if isinstance(batch, (list, tuple)):
        converted = [to_backend(v, dtype=dtype, wrap_tensors=wrap_tensors) for v in batch]
        return type(batch)(converted)

    # numpy scalar / python scalar
    if isinstance(batch, (int, float, bool)) or _is_np_scalar(batch):
        out = xp.asarray(batch, dtype=(dtype if dtype is not None else None))
        if wrap_tensors:
            out = Tensor(out, requires_grad=False)
        return out

    # Fallback: keep as-is
    return batch