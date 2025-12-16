import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor

xp = backend.xp


def _is_xp_array(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


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
    if _is_xp_array(out):
        return out.astype(xp.int64)
    return xp.asarray(out, dtype=xp.int64)


def _pad_1d(ids, max_len, pad_id=0, truncate=True):
    """
    ids: xp int64 1D
    returns (padded_ids, attn_mask, length)
    """
    if not _is_xp_array(ids):
        ids = xp.asarray(ids, dtype=xp.int64)
    ids = ids.astype(xp.int64)

    L = int(ids.shape[0])
    if truncate and L > max_len:
        ids = ids[:max_len]
        L = max_len

    out = xp.full((max_len,), int(pad_id), dtype=xp.int64)
    out[:L] = ids
    mask = xp.zeros((max_len,), dtype=xp.int64)
    mask[:L] = 1
    return out, mask, xp.asarray(L, dtype=xp.int64)


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