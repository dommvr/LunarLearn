import LunarLearn.core.backend.backend as backend
from .tensor import Tensor
from .parameter import Parameter
import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION
SAFE_FACTOR = backend.SAFE_FACTOR

if xp.__name__ == 'cupy':
    try:
        from cupyx.scatter_add import scatter_add
    except ImportError:
        from cupyx._scatter import scatter_add

def ensure_tensor(obj, dtype=None):
    """
    Ensure the input is a Tensor.
    Scalars, lists, numpy/cupy arrays get wrapped automatically.
    """
    if isinstance(obj, Tensor):
        return obj
    if isinstance(obj, Parameter):
        return obj.master
    data = xp.array(obj, dtype=DTYPE)
    return Tensor(data, dtype=dtype)

def normalize_index(idx):
    if isinstance(idx, Tensor):
        arr = idx.data
        if arr.shape == ():
            return int(arr)
        return arr
    
    if isinstance(idx, (tuple, list)):
        return tuple(normalize_index(i) for i in idx)
    
    return idx

def promote_dtype(*dtypes):
    """
    Promote multiple dtypes to a common dtype.

    Rules:
        - If any dtype is float32, result is float32.
        - Otherwise, result is float16.

    Args:
        *dtypes (str or Tensor): Dtypes or tensors whose dtypes should be promoted.

    Returns:
        str: Promoted dtype ('float16' or 'float32').
    """
    # Extract dtype strings if Tensor objects are passed
    dtype_list = []
    for d in dtypes:
        if hasattr(d, "dtype"):   # Tensor
            dtype_list.append(d.dtype)
        else:                     # Already a dtype string
            dtype_list.append(d)

    if "float32" in dtype_list:
        return "float32"
    return "float16"

def unbroadcast(grad, shape):
    """
    Reduce `grad` back to `shape` (the original tensor shape before broadcasting).
    """
    # Drop leading dims that got added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # For broadcasted axes (dim=1), sum along that axis
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

def trace_graph(tensor, depth=0, visited=None):
    if visited is None:
        visited = set()
    if tensor in visited:
        return
    visited.add(tensor)
    print("  " * depth + f"Tensor(id={id(tensor)}, grad_fn={tensor.grad_fn}, shape={tensor.shape})")
    for p in tensor._prev:
        trace_graph(p, depth + 1, visited)

def debug_topo(tensor: Tensor):
    """
    Prints the autograd graph in topological order.
    Shows each node's grad_fn and shape.
    """
    topo = []
    visited = set()

    def build(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)

    build(tensor)

    for node in topo:
        print(f"{node.grad_fn} -> {node.shape}")

def _any_requires_grad(args):
    for a in args:
        if isinstance(a, Tensor) and getattr(a, "requires_grad", False):
            return True
    return False

def checkpoint(fn, *args, **kwargs):
    """
    Memory-saving forward pass for a function `fn(*args, **kwargs)`.

    - Forward pass runs under no_grad() so intermediate activations are not stored.
    - During backward, forward is recomputed with grad enabled and gradients flow normally.

    Args:
        fn (callable): Function/layer to run with checkpointing.
        *args: Positional Tensor/non-Tensor arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor: A new Tensor whose backward pass recomputes the forward function.
    """
    # Fast path: nothing requires grad or global grad disabled → just run forward normally
    if not backend.is_grad_enabled() or not _any_requires_grad(args):
        return fn(*args, **kwargs)

    # 1) Forward pass under no_grad() so no intermediate graph is stored
    with backend.no_grad():
        detached_args = [
            a.detach() if isinstance(a, Tensor) else a
            for a in args
        ]
        out = fn(*detached_args, **kwargs)

    if not isinstance(out, Tensor):
        raise TypeError("checkpoint(fn, ...) currently supports a single Tensor output.")

    # 2) Wrap the output in a new Tensor with a custom backward hook
    out_cp = Tensor(out.data, requires_grad=True, dtype=out.dtype)
    out_cp.is_leaf = False
    out_cp.grad_fn = "checkpoint"

    saved_args = args  # original args (not detached)

    def _backward():
        if out_cp.grad is None:
            return

        # Recompute forward with grad enabled (this time tracking the graph)
        with backend.enable_grad():
            re_args = []
            for a in saved_args:
                if isinstance(a, Tensor):
                    # Create a new Tensor mirroring original requires_grad
                    re_args.append(a.clone())
                else:
                    re_args.append(a)

            re_out = fn(*re_args, **kwargs)
            if not isinstance(re_out, Tensor):
                raise TypeError("Recomputed output must be a Tensor.")

            # Instead of manually setting .grad, directly pass upstream grad into backward()
            re_out.backward(out_cp.grad)

            # Accumulate gradients back into original inputs
            for orig, replica in zip(saved_args, re_args):
                if isinstance(orig, Tensor) and getattr(orig, "requires_grad", False):
                    if replica.grad is not None:
                        if orig.grad is None:
                            orig.grad = replica.grad
                        else:
                            orig.grad += replica.grad

    out_cp._backward = _backward
    out_cp._prev = {a for a in args if isinstance(a, Tensor)}
    return out_cp


def _to_tuple(x, dim):
    if isinstance(x, int):
        return (x,) * dim
    if isinstance(x, (tuple, list)) and len(x) == dim:
        return tuple(int(v) for v in x)
    raise ValueError(f"Expected int or tuple/list of length {dim}, got {x!r}")


def _prod(xs):
    p = 1
    for v in xs:
        p *= int(v)
    return int(p)


def _scatter_add_flat(dst_flat, flat_idx, vals):
    # dst_flat: 1D
    if xp.__name__ == "cupy":
        scatter_add(dst_flat, (flat_idx,), vals)
    else:
        xp.add.at(dst_flat, flat_idx, vals)


def _safe_batch_size(X_shape, kernel_size, s, dilation=1, safety_factor=SAFE_FACTOR):
    """
    Estimate safe batch size based on free GPU memory for im2col.

    Supports:
      1D: X_shape = (m, C, L)
      2D: X_shape = (m, C, H, W)
      3D: X_shape = (m, C, D, H, W)

    kernel_size:
      - int (for 1D), or tuple/list length = spatial dims
    s (stride):
      - int or tuple/list length = spatial dims
    dilation:
      - int or tuple/list length = spatial dims

    Returns:
      batch_size: int in [1, m]
    """
    if len(X_shape) < 3 or len(X_shape) > 5:
        raise ValueError(f"Unsupported X_shape {X_shape}. Expected (m,C,L) / (m,C,H,W) / (m,C,D,H,W).")

    m = int(X_shape[0])
    C = int(X_shape[1])
    spatial = tuple(int(v) for v in X_shape[2:])
    dim = len(spatial)

    k = _to_tuple(kernel_size, dim)
    s = _to_tuple(s, dim)
    d = _to_tuple(dilation, dim)

    # Effective kernel per dim with dilation
    eff_k = [d[i] * (k[i] - 1) + 1 for i in range(dim)]

    # Output spatial sizes
    out_spatial = []
    for i in range(dim):
        out_i = (spatial[i] - eff_k[i]) // s[i] + 1
        # If out_i <= 0, im2col will be nonsense anyway, but don't crash here
        out_spatial.append(max(0, int(out_i)))

    # Elements produced by im2col per image:
    # (C * prod(kernel)) * prod(out_spatial)
    prod_k = 1
    for v in k:
        prod_k *= v

    prod_out = 1
    for v in out_spatial:
        prod_out *= v

    cols_per_image = C * prod_k * prod_out
    bytes_per_image = int(cols_per_image * xp.dtype(DTYPE).itemsize)

    # If bytes_per_image is 0 (e.g., invalid shapes), fall back to 1
    if bytes_per_image <= 0:
        return 1

    if xp.__name__ == "cupy":
        free_bytes, _ = xp.cuda.runtime.memGetInfo()
        avail = int(free_bytes * safety_factor)
    else:
        # CPU: assume we can handle full batch (or whatever the caller asked for)
        avail = bytes_per_image * m

    batch_size = max(1, min(m, avail // bytes_per_image))
    return int(batch_size)


# -------------------------
# 1D
# -------------------------
def _im2col1d_vectorized(X, kernel_size, s, dilation=1):
    # X: (m, C, L)
    m, n_C, n_L = X.shape
    kL = int(kernel_size)
    (sL,) = _to_tuple(s, 1)
    (dL,) = _to_tuple(dilation, 1)

    eff_kL = dL * (kL - 1) + 1
    out_L = (n_L - eff_kL) // sL + 1

    i0 = xp.tile(xp.arange(kL) * dL, n_C)                 # (C*kL,)
    i1 = sL * xp.arange(out_L)                            # (out_L,)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)            # (C*kL, out_L)

    k = xp.repeat(xp.arange(n_C), kL).reshape(-1, 1)      # (C*kL, 1)

    k = k.astype(xp.int32)
    i = i.astype(xp.int32)

    cols = X[:, k, i]                                     # (m, C*kL, out_L)
    cols = cols.transpose(1, 2, 0).reshape(n_C * kL, -1)  # (C*kL, m*out_L)
    return cols


def _im2col1d_safe_batch(X, kernel_size, s, dilation=1):
    # Same idea as your 2D safe batching: chunk along batch axis m
    m = X.shape[0]
    batch = _safe_batch_size(X.shape, kernel_size, s, dilation)
    out_list = []
    for start in range(0, m, batch):
        end = min(start + batch, m)
        out_list.append(_im2col1d_vectorized(X[start:end], kernel_size, s, dilation))
    return xp.concatenate(out_list, axis=1).astype(DTYPE)


def im2col1d(X, kernel_size, s, dilation=1):
    try:
        return _im2col1d_vectorized(X, kernel_size, s, dilation)
    except Exception:
        return _im2col1d_safe_batch(X, kernel_size, s, dilation)
    

def im2col1d_grouped(X, kernel_size, s, groups, dilation=1):
    """
    X: (m, C, L)
    Returns:
      cols shape (C/group * kL * groups, out_L * m)
    """
    m, C, L = X.shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    gc = C // groups
    cols_list = []
    for g in range(groups):
        start = g * gc
        end = (g + 1) * gc
        Xg = X[:, start:end, :]
        cols_list.append(im2col1d(Xg, kernel_size, s, dilation))
    return xp.concatenate(cols_list, axis=0)


def _col2im1d_vectorized(cols, X_shape, kernel_size, s, dilation=1):
    """
    Channel-first col2im1d: X_shape = (m, C, L)
    cols shape (C*kL, outL*m)
    """
    m, C, L = X_shape
    kL = int(kernel_size)
    (sL,) = _to_tuple(s, 1)
    (dL,) = _to_tuple(dilation, 1)

    eff_kL = dL * (kL - 1) + 1
    outL = (L - eff_kL) // sL + 1

    # indices like im2col1d
    i0 = xp.tile(xp.arange(kL) * dL, C)                  # (C*kL,)
    i1 = sL * xp.arange(outL)                             # (outL,)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)            # (C*kL, outL)
    k = xp.repeat(xp.arange(C), kL).reshape(-1, 1)        # (C*kL, 1)

    # reshape cols to (m, C*kL, outL)
    cols_r = cols.reshape(C * kL, outL, m).transpose(2, 0, 1)  # (m, C*kL, outL)

    # Flatten scatter indices into X_flat
    batch_idx = xp.repeat(xp.arange(m), i.size)           # m * (C*kL*outL)

    k_idx = xp.tile(xp.tile(k, (1, outL)).ravel(), m)     # repeated per batch
    i_idx = xp.tile(i, (m, 1)).ravel()

    batch_idx = batch_idx.astype(xp.int32)
    k_idx = k_idx.astype(xp.int32)
    i_idx = i_idx.astype(xp.int32)

    vals = cols_r.ravel()
    flat_idx = xp.ravel_multi_index((batch_idx, k_idx, i_idx), X_shape)

    X_flat = xp.zeros(m * C * L, dtype=cols.dtype)
    _scatter_add_flat(X_flat, flat_idx, vals)

    return X_flat.reshape(X_shape)


def _col2im1d_safe_batch(cols, X_shape, kernel_size, s, dilation=1):
    m, C, L = X_shape
    kL = int(kernel_size)

    batch = _safe_batch_size(X_shape, kernel_size=kL, s=s, dilation=dilation)
    (sL,) = _to_tuple(s, 1)
    (dL,) = _to_tuple(dilation, 1)

    eff_kL = dL * (kL - 1) + 1
    outL = (L - eff_kL) // sL + 1
    patches = outL

    X = xp.zeros(X_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        X_batch = _col2im1d_vectorized(cols_batch, (end - start, C, L), kL, s, dilation)
        X[start:end] = X_batch
    return X


def col2im1d(cols, X_shape, kernel_size, s, dilation=1):
    try:
        return _col2im1d_vectorized(cols, X_shape, kernel_size, s, dilation)
    except Exception:
        return _col2im1d_safe_batch(cols, X_shape, kernel_size, s, dilation)
    

def col2im1d_grouped(cols, X_shape, kernel_size, s, groups, dilation=1):
    m, C, L = X_shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    gc = C // groups
    rows_per_group = cols.shape[0] // groups

    X = xp.zeros(X_shape, dtype=cols.dtype)
    for g in range(groups):
        rs = g * rows_per_group
        re = (g + 1) * rows_per_group
        cs = g * gc
        ce = (g + 1) * gc
        cols_g = cols[rs:re, :]
        X[:, cs:ce, :] += col2im1d(cols_g, (m, gc, L), kernel_size, s, dilation)
    return X


def _im2col_transpose1d_vectorized(X, kernel_size, s, output_shape, padding=0, dilation=1):
    # X: (m, C, L) -> (C, L*m) with batch last (consistent with your reshape conventions)
    m, C, L = X.shape
    cols = X.transpose(1, 2, 0).reshape(C, -1)
    return cols


def im2col_transpose1d(X, kernel_size, s, output_shape, padding=0, dilation=1):
    return _im2col_transpose1d_vectorized(X, kernel_size, s, output_shape, padding, dilation)


def im2col_transpose1d_grouped(X, kernel_size, s, output_shape, groups, padding=0, dilation=1):
    m, C, L = X.shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")
    gc = C // groups
    cols_list = []
    for g in range(groups):
        cs, ce = g * gc, (g + 1) * gc
        cols_list.append(im2col_transpose1d(X[:, cs:ce, :], kernel_size, s, output_shape, padding, dilation))
    return xp.concatenate(cols_list, axis=0)


def _col2im_transpose1d_vectorized(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    """
    input_shape : (m, C_in, L_in)   (only m and L_in are used)
    output_shape: (m, C_out, L_out)
    cols shape  : (C_out*kL, L_in*m)
    returns     : (m, C_out, L_out)
    """
    m, _, L_in = input_shape
    m2, C_out, L_out = output_shape
    if m2 != m:
        raise ValueError("output_shape batch must match input_shape batch")

    kL = int(kernel_size)
    (sL,) = _to_tuple(s, 1)
    (pL,) = _to_tuple(padding, 1)
    (dL,) = _to_tuple(dilation, 1)

    # indices
    # kernel offsets
    i0 = xp.arange(kL) * dL                                # (kL,)
    # input positions
    x1 = xp.arange(L_in) * sL - pL                         # (L_in,)
    i = i0.reshape(-1, 1) + x1.reshape(1, -1)              # (kL, L_in)

    # replicate for channels
    i_full = xp.tile(i, (C_out, 1))                        # (C_out*kL, L_in)
    c = xp.repeat(xp.arange(C_out), kL).reshape(-1, 1)      # (C_out*kL, 1)

    rows = C_out * kL
    patches = L_in

    cols_r = cols.reshape(rows, patches, m).transpose(2, 0, 1)  # (m, rows, patches)
    vals = cols_r.ravel()

    # build indices in the same ravel order: batch -> row -> patch
    row_patch = xp.tile(c, (1, patches)).ravel()            # (rows*patches,)
    c_idx = xp.tile(row_patch, m)                           # (m*rows*patches,)

    i_idx = xp.tile(i_full.ravel(), m)                      # (m*rows*patches,)
    b_idx = xp.repeat(xp.arange(m), rows * patches)

    # mask valid
    valid = (i_idx >= 0) & (i_idx < L_out)
    if valid.ndim != 0:
        b_idx = b_idx[valid]
        c_idx = c_idx[valid]
        i_idx = i_idx[valid]
        vals  = vals[valid]

    b_idx = b_idx.astype(xp.int32)
    c_idx = c_idx.astype(xp.int32)
    i_idx = i_idx.astype(xp.int32)

    flat_idx = xp.ravel_multi_index((b_idx, c_idx, i_idx), output_shape)

    out_flat = xp.zeros(_prod(output_shape), dtype=cols.dtype)
    _scatter_add_flat(out_flat, flat_idx, vals)
    return out_flat.reshape(output_shape)


def _col2im_transpose1d_safe_batch(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    m, _, L_in = input_shape
    patches = L_in
    batch = _safe_batch_size(input_shape, kernel_size=1, s=1, dilation=1)  # conservative, cheap op

    out = xp.zeros(output_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        out_batch = _col2im_transpose1d_vectorized(
            cols_batch,
            (end - start, input_shape[1], L_in),
            kernel_size, s,
            (end - start, output_shape[1], output_shape[2]),
            padding, dilation
        )
        out[start:end] = out_batch
    return out


def col2im_transpose1d(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    try:
        return _col2im_transpose1d_vectorized(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
    except Exception:
        return _col2im_transpose1d_safe_batch(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
    

def col2im_transpose1d_grouped(cols, input_shape, kernel_size, s, output_shape, groups, padding=0, dilation=1):
    m, C_out, L_out = output_shape
    if C_out % groups != 0:
        raise ValueError(f"Output channels ({C_out}) must be divisible by groups ({groups})")
    kL = int(kernel_size)
    Cout_g = C_out // groups
    rows_per_group = Cout_g * kL

    out = xp.zeros(output_shape, dtype=cols.dtype)
    for g in range(groups):
        rs, re = g * rows_per_group, (g + 1) * rows_per_group
        cs, ce = g * Cout_g, (g + 1) * Cout_g
        cols_g = cols[rs:re, :]
        out[:, cs:ce, :] = col2im_transpose1d(cols_g, input_shape, kernel_size, s, (m, Cout_g, L_out), padding, dilation)
    return out


# -------------------------
# 2D
# -------------------------
def _im2col2d_vectorized(X, kernel_size, s, dilation=1):
    # X: (m, C, H, W)
    m, n_C, n_H, n_W = X.shape
    kH, kW = _to_tuple(kernel_size, 2)
    sH, sW = _to_tuple(s, 2)
    dH, dW = _to_tuple(dilation, 2)

    eff_kH = dH * (kH - 1) + 1
    eff_kW = dW * (kW - 1) + 1

    out_H = (n_H - eff_kH) // sH + 1
    out_W = (n_W - eff_kW) // sW + 1
    outN = out_H * out_W

    # Same index generation pattern as your original, generalized to tuple stride
    i0 = xp.repeat(xp.arange(kH) * dH, kW)          # (kH*kW,)
    i0 = xp.tile(i0, n_C)                            # (C*kH*kW,)
    i1 = sH * xp.repeat(xp.arange(out_H), out_W)     # (outN,)

    j0 = xp.tile(xp.arange(kW) * dW, kH * n_C)       # (C*kH*kW,)
    j1 = sW * xp.tile(xp.arange(out_W), out_H)       # (outN,)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)        # (C*kH*kW, outN)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)        # (C*kH*kW, outN)
    k = xp.repeat(xp.arange(n_C), kH * kW).reshape(-1, 1)  # (C*kH*kW, 1)

    k = k.astype(xp.int32)
    i = i.astype(xp.int32)
    j = j.astype(xp.int32)

    cols = X[:, k, i, j]                              # (m, C*kH*kW, outN)
    cols = cols.transpose(1, 2, 0).reshape(n_C * kH * kW, -1)  # (C*kH*kW, m*outN)
    return cols


def _im2col2d_safe_batch(X, kernel_size, s, dilation=1):
    m = X.shape[0]
    # IMPORTANT: pass dilation, and allow tuple kernel/stride
    batch = _safe_batch_size(X.shape, kernel_size=kernel_size, s=s, dilation=dilation)
    out_list = []
    for start in range(0, m, batch):
        end = min(start + batch, m)
        out_list.append(_im2col2d_vectorized(X[start:end], kernel_size, s, dilation))
    return xp.concatenate(out_list, axis=1).astype(DTYPE)


def im2col2d(X, kernel_size, s, dilation=1):
    # Normalize kernel_size to tuple(len=2) like 1D/3D style
    try:
        return _im2col2d_vectorized(X, kernel_size, s, dilation)
    except Exception:
        return _im2col2d_safe_batch(X, kernel_size, s, dilation)
    

def im2col2d_grouped(X, kernel_size, s, groups, dilation=1):
    """
    Group-aware im2col.

    Args:
        X: Input tensor (m, C, H, W)
        f: kernel size
        s: stride
        groups: number of groups (C must be divisible by groups)

    Returns:
        cols: im2col output with shape (C/group * f * f * groups, H_out * W_out * m)
    """
    m, C, H, W = X.shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    group_channels = C // groups
    cols_list = []

    # Split channel dimension and process each group
    for g in range(groups):
        start = g * group_channels
        end = (g + 1) * group_channels
        X_group = X[:, start:end, :, :]  # slice channels for this group
        cols_group = im2col2d(X_group, kernel_size, s, dilation)
        cols_list.append(cols_group)

    # Concatenate along the first dimension (channel*k*k axis)
    cols = xp.concatenate(cols_list, axis=0)
    return cols


def _col2im2d_vectorized(cols, X_shape, kernel_size, s, dilation=1):
    """
    Channel-first col2im2d: X_shape = (m, C, H, W)
    cols shape: (C*kH*kW, H_out*W_out*m)
    Consistent with col2im1d/col2im3d:
      - supports tuple/int stride
      - supports tuple/int dilation
      - uses effective kernel sizes
      - clean scatter-add indexing
    """
    m, C, H, W = X_shape
    kH, kW = _to_tuple(kernel_size, 2)
    sH, sW = _to_tuple(s, 2)
    dH, dW = _to_tuple(dilation, 2)

    eff_kH = dH * (kH - 1) + 1
    eff_kW = dW * (kW - 1) + 1

    H_out = (H - eff_kH) // sH + 1
    W_out = (W - eff_kW) // sW + 1
    outN = H_out * W_out

    # Same index generation scheme as im2col (generalized for tuple stride)
    i0 = xp.repeat(xp.arange(kH) * dH, kW)             # (kH*kW,)
    i0 = xp.tile(i0, C)                                # (C*kH*kW,)
    i1 = sH * xp.repeat(xp.arange(H_out), W_out)        # (outN,)

    j0 = xp.tile(xp.arange(kW) * dW, kH * C)            # (C*kH*kW,)
    j1 = sW * xp.tile(xp.arange(W_out), H_out)          # (outN,)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)           # (C*kH*kW, outN)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)           # (C*kH*kW, outN)
    k = xp.repeat(xp.arange(C), kH * kW).reshape(-1, 1)  # (C*kH*kW, 1)

    # Reshape cols -> (m, rows, outN)
    rows = C * kH * kW
    cols_r = cols.reshape(rows, outN, m).transpose(2, 0, 1)  # (m, rows, outN)

    # Build flat scatter indices
    batch_idx = xp.repeat(xp.arange(m), i.size)               # m * (rows*outN)

    k_idx = xp.tile(xp.tile(k, (1, outN)).ravel(), m)         # (m*rows*outN,)
    i_idx = xp.tile(i, (m, 1)).ravel()
    j_idx = xp.tile(j, (m, 1)).ravel()

    batch_idx = batch_idx.astype(xp.int32)
    k_idx = k_idx.astype(xp.int32)
    i_idx = i_idx.astype(xp.int32)
    j_idx = j_idx.astype(xp.int32)

    vals = cols_r.ravel()
    flat_idx = xp.ravel_multi_index((batch_idx, k_idx, i_idx, j_idx), X_shape)

    X_flat = xp.zeros(m * C * H * W, dtype=cols.dtype)
    _scatter_add_flat(X_flat, flat_idx, vals)
    return X_flat.reshape(X_shape)


def _col2im2d_safe_batch(cols, X_shape, kernel_size, s, dilation=1):
    m, C, H, W = X_shape

    # normalize kernel/stride/dilation like 1d/3d
    kH, kW = _to_tuple(kernel_size, 2)
    sH, sW = _to_tuple(s, 2)
    dH, dW = _to_tuple(dilation, 2)

    # IMPORTANT: include dilation in safe batch estimate
    batch = _safe_batch_size(X_shape, kernel_size=(kH, kW), s=(sH, sW), dilation=(dH, dW))

    # effective kernel sizes
    eff_kH = dH * (kH - 1) + 1
    eff_kW = dW * (kW - 1) + 1

    # output spatial sizes
    H_out = (H - eff_kH) // sH + 1
    W_out = (W - eff_kW) // sW + 1
    patches = H_out * W_out

    X = xp.zeros(X_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        X_batch = _col2im2d_vectorized(cols_batch, (end - start, C, H, W), (kH, kW), (sH, sW), (dH, dW))
        X[start:end] = X_batch
    return X


def col2im2d(cols, X_shape, kernel_size, s, dilation=1):
    try:
        return _col2im2d_vectorized(cols, X_shape, kernel_size, s, dilation)
    except Exception:
        return _col2im2d_safe_batch(cols, X_shape, kernel_size, s, dilation)


def col2im2d_grouped(cols, X_shape, kernel_size, s, groups, dilation=1):
    """
    Group-aware col2im.

    Args:
        cols: im2col-style matrix from grouped convolution
        X_shape: original input shape (m, C, H, W)
        f: kernel size
        s: stride
        groups: number of groups

    Returns:
        X_reconstructed: Reconstructed input tensor (m, C, H, W)
    """
    m, C, H, W = X_shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    group_channels = C // groups
    cols_per_group = cols.shape[0] // groups

    X_reconstructed = xp.zeros(X_shape, dtype=cols.dtype)

    # Split cols and scatter for each group
    for g in range(groups):
        start_cols = g * cols_per_group
        end_cols = (g + 1) * cols_per_group
        start_ch = g * group_channels
        end_ch = (g + 1) * group_channels

        cols_group = cols[start_cols:end_cols, :]
        X_reconstructed[:, start_ch:end_ch, :, :] += col2im2d(cols_group, (m, group_channels, H, W), kernel_size, s, dilation)

    return X_reconstructed


def _im2col_transpose2d_vectorized(X, kernel_size, s, output_shape, padding=0, dilation=1):
    # X: (m, C, H, W) -> (C, H*W*m)
    m, C, H, W = X.shape
    cols = X.transpose(1, 2, 3, 0).reshape(C, -1)
    return cols


def im2col_transpose2d(X, kernel_size, s, output_shape, padding=0, dilation=1):
    return _im2col_transpose2d_vectorized(X, kernel_size, s, output_shape, padding, dilation)
    

def im2col_transpose2d_grouped(X, kernel_size, s, output_shape, groups, padding=0, dilation=1):
    m, C, H, W = X.shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")
    gc = C // groups
    cols_list = []
    for g in range(groups):
        cs, ce = g * gc, (g + 1) * gc
        cols_list.append(im2col_transpose2d(X[:, cs:ce, :, :], kernel_size, s, output_shape, padding, dilation))
    return xp.concatenate(cols_list, axis=0)


def _col2im_transpose2d_vectorized(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    """
    input_shape : (m, C_in, H_in, W_in)
    output_shape: (m, C_out, H_out, W_out)
    cols shape  : (C_out*kH*kW, (H_in*W_in)*m)
    returns     : (m, C_out, H_out, W_out)
    """
    m, _, H_in, W_in = input_shape
    m2, C_out, H_out, W_out = output_shape
    if m2 != m:
        raise ValueError("output_shape batch must match input_shape batch")

    kH, kW = _to_tuple(kernel_size, 2)
    sH, sW = _to_tuple(s, 2)
    pH, pW = _to_tuple(padding, 2)
    dH, dW = _to_tuple(dilation, 2)

    # kernel offsets
    i0 = xp.repeat(xp.arange(kH) * dH, kW)                  # (kH*kW,)
    j0 = xp.tile(xp.arange(kW) * dW, kH)                    # (kH*kW,)

    # input positions (flattened)
    h_in = xp.repeat(xp.arange(H_in), W_in)                 # (H_in*W_in,)
    w_in = xp.tile(xp.arange(W_in), H_in)                   # (H_in*W_in,)

    i1 = h_in * sH - pH
    j1 = w_in * sW - pW

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)               # (kH*kW, patches)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)               # (kH*kW, patches)

    kK = kH * kW
    patches = H_in * W_in
    rows = C_out * kK

    # replicate for channels
    i_full = xp.tile(i, (C_out, 1))                         # (rows, patches)
    j_full = xp.tile(j, (C_out, 1))                         # (rows, patches)
    c = xp.repeat(xp.arange(C_out), kK).reshape(-1, 1)       # (rows, 1)

    cols_r = cols.reshape(rows, patches, m).transpose(2, 0, 1)  # (m, rows, patches)
    vals = cols_r.ravel()

    row_patch_c = xp.tile(c, (1, patches)).ravel()           # (rows*patches,)
    c_idx = xp.tile(row_patch_c, m)

    i_idx = xp.tile(i_full.ravel(), m)
    j_idx = xp.tile(j_full.ravel(), m)
    b_idx = xp.repeat(xp.arange(m), rows * patches)

    valid = (i_idx >= 0) & (i_idx < H_out) & (j_idx >= 0) & (j_idx < W_out)
    if valid.ndim != 0:
        b_idx = b_idx[valid]
        c_idx = c_idx[valid]
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]
        vals  = vals[valid]

    b_idx = b_idx.astype(xp.int32)
    c_idx = c_idx.astype(xp.int32)
    i_idx = i_idx.astype(xp.int32)
    j_idx = j_idx.astype(xp.int32)

    flat_idx = xp.ravel_multi_index((b_idx, c_idx, i_idx, j_idx), output_shape)

    out_flat = xp.zeros(_prod(output_shape), dtype=cols.dtype)
    _scatter_add_flat(out_flat, flat_idx, vals)
    return out_flat.reshape(output_shape)


def _col2im_transpose2d_safe_batch(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    m, _, H_in, W_in = input_shape
    patches = H_in * W_in
    batch = _safe_batch_size(input_shape, kernel_size=(1, 1), s=(1, 1), dilation=(1, 1))  # conservative

    out = xp.zeros(output_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        out_batch = _col2im_transpose2d_vectorized(
            cols_batch,
            (end - start, input_shape[1], H_in, W_in),
            kernel_size, s,
            (end - start, output_shape[1], output_shape[2], output_shape[3]),
            padding, dilation
        )
        out[start:end] = out_batch
    return out


def col2im_transpose2d(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    try:
        return _col2im_transpose2d_vectorized(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
    except Exception:
        return _col2im_transpose2d_safe_batch(cols, input_shape, kernel_size, s, output_shape, padding, dilation)


def col2im_transpose2d_grouped(cols, input_shape, kernel_size, s, output_shape, groups, padding=0, dilation=1):
    m, C_out, H_out, W_out = output_shape
    if C_out % groups != 0:
        raise ValueError(f"Output channels ({C_out}) must be divisible by groups ({groups})")
    kH, kW = _to_tuple(kernel_size, 2)
    Cout_g = C_out // groups
    rows_per_group = Cout_g * (kH * kW)

    out = xp.zeros(output_shape, dtype=cols.dtype)
    for g in range(groups):
        rs, re = g * rows_per_group, (g + 1) * rows_per_group
        cs, ce = g * Cout_g, (g + 1) * Cout_g
        cols_g = cols[rs:re, :]
        out[:, cs:ce, :, :] = col2im_transpose2d(cols_g, input_shape, kernel_size, s, (m, Cout_g, H_out, W_out), padding, dilation)
    return out


# -------------------------
# 3D
# -------------------------
def _im2col3d_vectorized(X, kernel_size, s, dilation=1):
    # X: (m, C, D, H, W)
    m, n_C, n_D, n_H, n_W = X.shape

    kD, kH, kW = _to_tuple(kernel_size, 3)
    sD, sH, sW = _to_tuple(s, 3)
    dD, dH, dW = _to_tuple(dilation, 3)

    eff_kD = dD * (kD - 1) + 1
    eff_kH = dH * (kH - 1) + 1
    eff_kW = dW * (kW - 1) + 1

    out_D = (n_D - eff_kD) // sD + 1
    out_H = (n_H - eff_kH) // sH + 1
    out_W = (n_W - eff_kW) // sW + 1

    # Kernel offsets (per channel)
    # Build (kD*kH*kW,) offsets, then tile over channels
    d0 = xp.repeat(xp.arange(kD) * dD, kH * kW)                  # (kD*kH*kW,)
    h0 = xp.tile(xp.repeat(xp.arange(kH) * dH, kW), kD)          # (kD*kH*kW,)
    w0 = xp.tile(xp.arange(kW) * dW, kD * kH)                    # (kD*kH*kW,)

    d0 = xp.tile(d0, n_C)                                        # (C*kD*kH*kW,)
    h0 = xp.tile(h0, n_C)
    w0 = xp.tile(w0, n_C)

    # Output positions
    d1 = sD * xp.repeat(xp.arange(out_D), out_H * out_W)         # (out_D*out_H*out_W,)
    h1 = sH * xp.tile(xp.repeat(xp.arange(out_H), out_W), out_D)
    w1 = sW * xp.tile(xp.arange(out_W), out_D * out_H)

    dd = d0.reshape(-1, 1) + d1.reshape(1, -1)                   # (C*kD*kH*kW, outN)
    hh = h0.reshape(-1, 1) + h1.reshape(1, -1)
    ww = w0.reshape(-1, 1) + w1.reshape(1, -1)

    k = xp.repeat(xp.arange(n_C), kD * kH * kW).reshape(-1, 1)   # (C*kD*kH*kW, 1)

    k = k.astype(xp.int32)
    dd = dd.astype(xp.int32)
    hh = hh.astype(xp.int32)
    ww = ww.astype(xp.int32)

    # Fancy indexing: result (m, C*kD*kH*kW, outN)
    cols = X[:, k, dd, hh, ww]
    cols = cols.transpose(1, 2, 0).reshape(n_C * kD * kH * kW, -1)  # (C*kD*kH*kW, m*outN)
    return cols


def _im2col3d_safe_batch(X, kernel_size, s, dilation=1):
    m = X.shape[0]
    batch = _safe_batch_size(X.shape, kernel_size, s, dilation)
    out_list = []
    for start in range(0, m, batch):
        end = min(start + batch, m)
        out_list.append(_im2col3d_vectorized(X[start:end], kernel_size, s, dilation))
    return xp.concatenate(out_list, axis=1).astype(DTYPE)


def im2col3d(X, kernel_size, s, dilation=1):
    try:
        return _im2col3d_vectorized(X, kernel_size, s, dilation)
    except Exception:
        return _im2col3d_safe_batch(X, kernel_size, s, dilation)
    

def im2col3d_grouped(X, kernel_size, s, groups, dilation=1):
    """
    X: (m, C, D, H, W)
    Returns:
      cols shape (C/group * kD*kH*kW * groups, out_D*out_H*out_W * m)
    """
    m, C, D, H, W = X.shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    gc = C // groups
    cols_list = []
    for g in range(groups):
        start = g * gc
        end = (g + 1) * gc
        Xg = X[:, start:end, :, :, :]
        cols_list.append(im2col3d(Xg, kernel_size, s, dilation))
    return xp.concatenate(cols_list, axis=0)


def _col2im3d_vectorized(cols, X_shape, kernel_size, s, dilation=1):
    """
    Channel-first col2im3d: X_shape = (m, C, D, H, W)
    cols shape (C*kD*kH*kW, outD*outH*outW*m)
    """
    m, C, D, H, W = X_shape
    kD, kH, kW = _to_tuple(kernel_size, 3)
    sD, sH, sW = _to_tuple(s, 3)
    dD, dH, dW = _to_tuple(dilation, 3)

    eff_kD = dD * (kD - 1) + 1
    eff_kH = dH * (kH - 1) + 1
    eff_kW = dW * (kW - 1) + 1

    outD = (D - eff_kD) // sD + 1
    outH = (H - eff_kH) // sH + 1
    outW = (W - eff_kW) // sW + 1
    outN = outD * outH * outW

    # same indices as im2col3d
    d0 = xp.repeat(xp.arange(kD) * dD, kH * kW)
    h0 = xp.tile(xp.repeat(xp.arange(kH) * dH, kW), kD)
    w0 = xp.tile(xp.arange(kW) * dW, kD * kH)

    d0 = xp.tile(d0, C)
    h0 = xp.tile(h0, C)
    w0 = xp.tile(w0, C)

    d1 = sD * xp.repeat(xp.arange(outD), outH * outW)
    h1 = sH * xp.tile(xp.repeat(xp.arange(outH), outW), outD)
    w1 = sW * xp.tile(xp.arange(outW), outD * outH)

    dd = d0.reshape(-1, 1) + d1.reshape(1, -1)          # (C*kD*kH*kW, outN)
    hh = h0.reshape(-1, 1) + h1.reshape(1, -1)
    ww = w0.reshape(-1, 1) + w1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), kD * kH * kW).reshape(-1, 1)

    cols_r = cols.reshape(C * kD * kH * kW, outN, m).transpose(2, 0, 1)  # (m, rows, outN)

    batch_idx = xp.repeat(xp.arange(m), dd.size)
    k_idx = xp.tile(xp.tile(k, (1, outN)).ravel(), m)
    d_idx = xp.tile(dd, (m, 1)).ravel()
    h_idx = xp.tile(hh, (m, 1)).ravel()
    w_idx = xp.tile(ww, (m, 1)).ravel()

    batch_idx = batch_idx.astype(xp.int32)
    k_idx = k_idx.astype(xp.int32)
    d_idx = d_idx.astype(xp.int32)
    h_idx = h_idx.astype(xp.int32)
    w_idx = w_idx.astype(xp.int32)

    vals = cols_r.ravel()
    flat_idx = xp.ravel_multi_index((batch_idx, k_idx, d_idx, h_idx, w_idx), X_shape)

    X_flat = xp.zeros(m * C * D * H * W, dtype=cols.dtype)
    _scatter_add_flat(X_flat, flat_idx, vals)
    return X_flat.reshape(X_shape)


def _col2im3d_safe_batch(cols, X_shape, kernel_size, s, dilation=1):
    m, C, D, H, W = X_shape
    batch = _safe_batch_size(X_shape, kernel_size=kernel_size, s=s, dilation=dilation)

    kD, kH, kW = _to_tuple(kernel_size, 3)
    sD, sH, sW = _to_tuple(s, 3)
    dD, dH, dW = _to_tuple(dilation, 3)

    eff_kD = dD * (kD - 1) + 1
    eff_kH = dH * (kH - 1) + 1
    eff_kW = dW * (kW - 1) + 1

    outD = (D - eff_kD) // sD + 1
    outH = (H - eff_kH) // sH + 1
    outW = (W - eff_kW) // sW + 1
    patches = outD * outH * outW

    X = xp.zeros(X_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        X_batch = _col2im3d_vectorized(cols_batch, (end - start, C, D, H, W), kernel_size, s, dilation)
        X[start:end] = X_batch
    return X


def col2im3d(cols, X_shape, kernel_size, s, dilation=1):
    try:
        return _col2im3d_vectorized(cols, X_shape, kernel_size, s, dilation)
    except Exception:
        return _col2im3d_safe_batch(cols, X_shape, kernel_size, s, dilation)
    

def col2im3d_grouped(cols, X_shape, kernel_size, s, groups, dilation=1):
    m, C, D, H, W = X_shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    gc = C // groups
    rows_per_group = cols.shape[0] // groups

    X = xp.zeros(X_shape, dtype=cols.dtype)
    for g in range(groups):
        rs = g * rows_per_group
        re = (g + 1) * rows_per_group
        cs = g * gc
        ce = (g + 1) * gc
        cols_g = cols[rs:re, :]
        X[:, cs:ce, :, :, :] += col2im3d(cols_g, (m, gc, D, H, W), kernel_size, s, dilation)
    return X


def _im2col_transpose3d_vectorized(X, kernel_size, s, output_shape, padding=0, dilation=1):
    # X: (m, C, D, H, W) -> (C, D*H*W*m)
    m, C, D, H, W = X.shape
    cols = X.transpose(1, 2, 3, 4, 0).reshape(C, -1)
    return cols


def im2col_transpose3d(X, kernel_size, s, output_shape, padding=0, dilation=1):
    return _im2col_transpose3d_vectorized(X, kernel_size, s, output_shape, padding, dilation)


def im2col_transpose3d_grouped(X, kernel_size, s, output_shape, groups, padding=0, dilation=1):
    m, C, D, H, W = X.shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")
    gc = C // groups
    cols_list = []
    for g in range(groups):
        cs, ce = g * gc, (g + 1) * gc
        cols_list.append(im2col_transpose3d(X[:, cs:ce, :, :, :], kernel_size, s, output_shape, padding, dilation))
    return xp.concatenate(cols_list, axis=0)


def _col2im_transpose3d_vectorized(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    """
    input_shape : (m, C_in, D_in, H_in, W_in)
    output_shape: (m, C_out, D_out, H_out, W_out)
    cols shape  : (C_out*kD*kH*kW, (D_in*H_in*W_in)*m)
    returns     : (m, C_out, D_out, H_out, W_out)
    """
    m, _, D_in, H_in, W_in = input_shape
    m2, C_out, D_out, H_out, W_out = output_shape
    if m2 != m:
        raise ValueError("output_shape batch must match input_shape batch")

    kD, kH, kW = _to_tuple(kernel_size, 3)
    sD, sH, sW = _to_tuple(s, 3)
    pD, pH, pW = _to_tuple(padding, 3)
    dD, dH, dW = _to_tuple(dilation, 3)

    # kernel offsets
    d0 = xp.repeat(xp.arange(kD) * dD, kH * kW)                     # (kD*kH*kW,)
    h0 = xp.tile(xp.repeat(xp.arange(kH) * dH, kW), kD)             # (kD*kH*kW,)
    w0 = xp.tile(xp.arange(kW) * dW, kD * kH)                       # (kD*kH*kW,)

    # input positions (flattened)
    # order: d major, then h, then w
    d_in = xp.repeat(xp.arange(D_in), H_in * W_in)
    h_in = xp.tile(xp.repeat(xp.arange(H_in), W_in), D_in)
    w_in = xp.tile(xp.arange(W_in), D_in * H_in)

    d1 = d_in * sD - pD
    h1 = h_in * sH - pH
    w1 = w_in * sW - pW

    dd = d0.reshape(-1, 1) + d1.reshape(1, -1)                      # (kK, patches)
    hh = h0.reshape(-1, 1) + h1.reshape(1, -1)
    ww = w0.reshape(-1, 1) + w1.reshape(1, -1)

    kK = kD * kH * kW
    patches = D_in * H_in * W_in
    rows = C_out * kK

    dd_full = xp.tile(dd, (C_out, 1))
    hh_full = xp.tile(hh, (C_out, 1))
    ww_full = xp.tile(ww, (C_out, 1))
    c = xp.repeat(xp.arange(C_out), kK).reshape(-1, 1)

    cols_r = cols.reshape(rows, patches, m).transpose(2, 0, 1)      # (m, rows, patches)
    vals = cols_r.ravel()

    row_patch_c = xp.tile(c, (1, patches)).ravel()
    c_idx = xp.tile(row_patch_c, m)

    d_idx = xp.tile(dd_full.ravel(), m)
    h_idx = xp.tile(hh_full.ravel(), m)
    w_idx = xp.tile(ww_full.ravel(), m)
    b_idx = xp.repeat(xp.arange(m), rows * patches)

    valid = (
        (d_idx >= 0) & (d_idx < D_out) &
        (h_idx >= 0) & (h_idx < H_out) &
        (w_idx >= 0) & (w_idx < W_out)
    )
    if valid.ndim != 0:
        b_idx = b_idx[valid]
        c_idx = c_idx[valid]
        d_idx = d_idx[valid]
        h_idx = h_idx[valid]
        w_idx = w_idx[valid]
        vals  = vals[valid]

    b_idx = b_idx.astype(xp.int32)
    c_idx = c_idx.astype(xp.int32)
    d_idx = d_idx.astype(xp.int32)
    h_idx = h_idx.astype(xp.int32)
    w_idx = w_idx.astype(xp.int32)

    flat_idx = xp.ravel_multi_index((b_idx, c_idx, d_idx, h_idx, w_idx), output_shape)

    out_flat = xp.zeros(_prod(output_shape), dtype=cols.dtype)
    _scatter_add_flat(out_flat, flat_idx, vals)
    return out_flat.reshape(output_shape)


def _col2im_transpose3d_safe_batch(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    m, _, D_in, H_in, W_in = input_shape
    patches = D_in * H_in * W_in
    batch = _safe_batch_size(input_shape, kernel_size=(1, 1, 1), s=(1, 1, 1), dilation=(1, 1, 1))  # conservative

    out = xp.zeros(output_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        out_batch = _col2im_transpose3d_vectorized(
            cols_batch,
            (end - start, input_shape[1], D_in, H_in, W_in),
            kernel_size, s,
            (end - start, output_shape[1], output_shape[2], output_shape[3], output_shape[4]),
            padding, dilation
        )
        out[start:end] = out_batch
    return out


def col2im_transpose3d(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    try:
        return _col2im_transpose3d_vectorized(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
    except Exception:
        return _col2im_transpose3d_safe_batch(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
    

def col2im_transpose3d_grouped(cols, input_shape, kernel_size, s, output_shape, groups, padding=0, dilation=1):
    m, C_out, D_out, H_out, W_out = output_shape
    if C_out % groups != 0:
        raise ValueError(f"Output channels ({C_out}) must be divisible by groups ({groups})")
    kD, kH, kW = _to_tuple(kernel_size, 3)
    Cout_g = C_out // groups
    rows_per_group = Cout_g * (kD * kH * kW)

    out = xp.zeros(output_shape, dtype=cols.dtype)
    for g in range(groups):
        rs, re = g * rows_per_group, (g + 1) * rows_per_group
        cs, ce = g * Cout_g, (g + 1) * Cout_g
        cols_g = cols[rs:re, :]
        out[:, cs:ce, :, :, :] = col2im_transpose3d(cols_g, input_shape, kernel_size, s, (m, Cout_g, D_out, H_out, W_out), padding, dilation)
    return out


# -------------------------
# Dispatcher
# -------------------------
def im2col(X, kernel_size, s, dilation=1, groups=1):
    """
    Dispatch based on X.ndim:
      3 -> 1D (m,C,L)
      4 -> 2D (m,C,H,W)
      5 -> 3D (m,C,D,H,W)

    If groups > 1: slices channels into groups, runs per-group im2col, concatenates along row axis.
    Output layout matches your 2D convention:
      rows = C * prod(kernel)
      cols = m * prod(out_spatial)
    """
    if groups < 1 or not isinstance(groups, int):
        raise ValueError("groups must be a positive integer")

    if X.ndim == 3:
        if groups == 1:
            return im2col1d(X, kernel_size, s, dilation)
        return im2col1d_grouped(X, kernel_size, s, groups, dilation)

    if X.ndim == 4:
        if groups == 1:
            return im2col2d(X, kernel_size, s, dilation)
        return im2col2d_grouped(X, kernel_size, s, groups, dilation)

    if X.ndim == 5:
        if groups == 1:
            return im2col3d(X, kernel_size, s, dilation)
        return im2col3d_grouped(X, kernel_size, s, groups, dilation)

    raise ValueError(f"Unsupported X.ndim={X.ndim}. Expected 3, 4, or 5.")


def col2im(cols, X_shape, kernel_size, s, dilation=1, groups=1):
    """
    Dispatch based on X_shape rank:
      3 -> 1D: (m,C,L)
      4 -> 2D: (m,C,H,W)
      5 -> 3D: (m,C,D,H,W)

    If groups > 1, assumes 'grouped cols' layout:
      rows concatenated by group, same as your im2col_grouped / im2col1d_grouped / im2col3d_grouped.
    """
    if groups < 1 or not isinstance(groups, int):
        raise ValueError("groups must be a positive integer")

    ndim = len(X_shape)
    if ndim == 3:
        if groups == 1:
            return col2im1d(cols, X_shape, kernel_size, s, dilation)
        return col2im1d_grouped(cols, X_shape, kernel_size, s, groups, dilation)

    if ndim == 4:
        if groups == 1:
            return col2im2d(cols, X_shape, kernel_size, s, dilation)
        return col2im2d_grouped(cols, X_shape, kernel_size, s, groups, dilation)

    if ndim == 5:
        if groups == 1:
            return col2im3d(cols, X_shape, kernel_size, s, dilation)
        return col2im3d_grouped(cols, X_shape, kernel_size, s, groups, dilation)

    raise ValueError(f"Unsupported X_shape rank {ndim}. Expected 3, 4, or 5.")


def im2col_transpose(X, kernel_size, s, output_shape, padding=0, dilation=1, groups=1):
    """
    Dispatch by X.ndim:
      3 -> 1D: (m,C,L)
      4 -> 2D: (m,C,H,W)
      5 -> 3D: (m,C,D,H,W)

    Output layout:
      (C, prod(in_spatial) * m) with batch last in the flatten order,
      consistent with your im2col/col2im reshape conventions.
    """
    if groups < 1 or not isinstance(groups, int):
        raise ValueError("groups must be a positive integer")

    if X.ndim == 3:
        if groups == 1:
            return im2col_transpose1d(X, kernel_size, s, output_shape, padding, dilation)
        return im2col_transpose1d_grouped(X, kernel_size, s, output_shape, groups, padding, dilation)

    if X.ndim == 4:
        if groups == 1:
            return im2col_transpose2d(X, kernel_size, s, output_shape, padding, dilation)
        return im2col_transpose2d_grouped(X, kernel_size, s, output_shape, groups, padding, dilation)

    if X.ndim == 5:
        if groups == 1:
            return im2col_transpose3d(X, kernel_size, s, output_shape, padding, dilation)
        return im2col_transpose3d_grouped(X, kernel_size, s, output_shape, groups, padding, dilation)

    raise ValueError(f"Unsupported X.ndim={X.ndim}. Expected 3, 4, or 5.")


def col2im_transpose(cols, input_shape, kernel_size, s, output_shape, padding=0, dilation=1, groups=1):
    """
    Dispatch by input_shape rank:
      3 -> 1D: (m,C,L)
      4 -> 2D: (m,C,H,W)
      5 -> 3D: (m,C,D,H,W)

    cols shape must be:
      (C_out * prod(kernel), prod(in_spatial) * m)
    """
    if groups < 1 or not isinstance(groups, int):
        raise ValueError("groups must be a positive integer")

    ndim = len(input_shape)
    if ndim == 3:
        if groups == 1:
            return col2im_transpose1d(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
        return col2im_transpose1d_grouped(cols, input_shape, kernel_size, s, output_shape, groups, padding, dilation)

    if ndim == 4:
        if groups == 1:
            return col2im_transpose2d(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
        return col2im_transpose2d_grouped(cols, input_shape, kernel_size, s, output_shape, groups, padding, dilation)

    if ndim == 5:
        if groups == 1:
            return col2im_transpose3d(cols, input_shape, kernel_size, s, output_shape, padding, dilation)
        return col2im_transpose3d_grouped(cols, input_shape, kernel_size, s, output_shape, groups, padding, dilation)

    raise ValueError(f"Unsupported input_shape rank {ndim}. Expected 3, 4, or 5.")


def im2col_transpose_grad_from_dout_2d(dout, input_shape, kernel_size, s, output_shape, padding=0, dilation=1):
    """
    This is the adjoint of _col2im_transpose2d_vectorized wrt cols.

    dout: (m, C_out, H_out, W_out)
    returns dcols: (C_out*kH*kW, (H_in*W_in)*m)
    """
    m, _, H_in, W_in = input_shape
    m2, C_out, H_out, W_out = output_shape
    if m2 != m:
        raise ValueError("output_shape batch must match input_shape batch")

    kH, kW = _to_tuple(kernel_size, 2)
    sH, sW = _to_tuple(s, 2)
    pH, pW = _to_tuple(padding, 2)
    dH, dW = _to_tuple(dilation, 2)

    # kernel offsets
    i0 = xp.repeat(xp.arange(kH) * dH, kW)       # (kK,)
    j0 = xp.tile(xp.arange(kW) * dW, kH)         # (kK,)

    # input positions (flattened patches)
    h_in = xp.repeat(xp.arange(H_in), W_in)      # (patches,)
    w_in = xp.tile(xp.arange(W_in), H_in)        # (patches,)

    i1 = h_in * sH - pH
    j1 = w_in * sW - pW

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)    # (kK, patches)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)    # (kK, patches)

    kK = kH * kW
    patches = H_in * W_in
    rows = C_out * kK

    # replicate for channels to match forward indexing
    i_full = xp.tile(i, (C_out, 1))              # (rows, patches)
    j_full = xp.tile(j, (C_out, 1))              # (rows, patches)
    c = xp.repeat(xp.arange(C_out), kK).reshape(-1, 1)  # (rows,1)

    # build gather indices in the same order as forward ravel: batch -> rows -> patches
    b_idx = xp.repeat(xp.arange(m), rows * patches)

    c_idx = xp.tile(xp.tile(c, (1, patches)).ravel(), m)         # (m*rows*patches,)
    i_idx = xp.tile(i_full.ravel(), m)
    j_idx = xp.tile(j_full.ravel(), m)

    valid = (i_idx >= 0) & (i_idx < H_out) & (j_idx >= 0) & (j_idx < W_out)

    # gather, invalid positions contribute 0
    vals = xp.zeros_like(i_idx, dtype=dout.dtype)
    if xp.any(valid):
        bv = b_idx[valid].astype(xp.int32)
        cv = c_idx[valid].astype(xp.int32)
        iv = i_idx[valid].astype(xp.int32)
        jv = j_idx[valid].astype(xp.int32)
        vals_valid = dout[bv, cv, iv, jv]
        vals[valid] = vals_valid

    # reshape back to (m, rows, patches) then to (rows, patches, m) and flatten to (rows, patches*m)
    vals = vals.reshape(m, rows, patches)
    dcols = vals.transpose(1, 2, 0).reshape(rows, -1)  # (rows, patches*m)
    return dcols


def upsample(x, scale_factor, mode, align_corners):
    if mode == "nearest":
        # Repeat pixels: H then W
        data = xp.repeat(x, scale_factor, axis=2)
        data = xp.repeat(data, scale_factor, axis=3)
    elif mode == "bilinear":
        if xp.__name__ == "cupy":
            import cupy
            data = cupy.ndimage.zoom(
                x, (1, 1, scale_factor, scale_factor),
                order=1, grid_mode=align_corners
            )
        elif xp.__name__ == "numpy":
            from scipy.ndimage import zoom
            data = zoom(
                x, (1, 1, scale_factor, scale_factor),
                order=1, grid_mode=align_corners
            )
        else:
            raise NotImplementedError(f"upsample not supported for {xp.__name__}")
    else:
        raise ValueError("mode must be 'bilinear' or 'nearest'")

    return data

class RecurrentDropout:
    """
    Recurrent dropout with fixed mask across timesteps.

    During training, generates a single mask of shape (1, batch_size, *feature_dims)
    and reuses it for every timestep. During evaluation, acts as identity.

    Usage:
        dropout = RecurrentDropout((batch_size, hidden_size), keep_prob, training=True)
        for t in range(T):
            h = dropout(h)  # same mask every step
    """
    def __init__(self, mask_shape: tuple[int, ...], keep_prob: float, training: bool = True):
        self.training = training
        dtype = C_DTYPE if MIXED_PRECISION else DTYPE

        if self.training and 0 < keep_prob < 1.0:
            mask = (xp.random.rand(*mask_shape) < keep_prob).astype(dtype)
            mask = Tensor(mask, requires_grad=False, dtype=dtype)

            # Scaling factor
            scale = Tensor(1.0 / keep_prob, requires_grad=False, dtype=dtype)
        elif self.training and keep_prob < 0:
            mask = ops.zeros(mask_shape, dtype=dtype)
            scale = ops.zeros((), dtype=dtype)
        else:
            mask = None
            scale = None

        self.mask = mask
        self.scale = scale

    def __call__(self, a: Tensor):
        if self.mask is not None:
            return a * self.mask * self.scale
        else:
            return a