import numpy as np
import cupy as cp

import LunarLearn.core.backend.backend as backend
from LunarLearn.data.dataloader import SubsetDataset
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