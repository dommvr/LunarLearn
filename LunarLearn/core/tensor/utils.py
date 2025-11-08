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

def _safe_batch_size(X_shape, kernel_size, s, safety_factor=SAFE_FACTOR):
    """
    Estimate safe batch size based on GPU free memory.
    X_shape expected NCHW: (m, C, H, W)
    """
    m, C, H, W = X_shape
    f_h, f_w = kernel_size
    H_out = (H - f_h) // s + 1
    W_out = (W - f_w) // s + 1

    # number of elements produced by im2col per image
    cols_per_image = f_h * f_w * C * H_out * W_out
    bytes_per_image = int(cols_per_image * xp.dtype(DTYPE).itemsize)

    if xp.__name__ == "cupy":
        free_bytes, _ = xp.cuda.runtime.memGetInfo()
        avail = int(free_bytes * safety_factor)
    else:
        # on CPU assume we can take whole dataset (or a large default)
        avail = bytes_per_image * m  # allow full batch

    batch_size = max(1, min(m, avail // max(1, bytes_per_image)))
    return batch_size

def _im2col_vectorized(X, kernel_size, s):
    m, n_C, n_H, n_W = X.shape
    f_h, f_w = kernel_size
    n_H_out = (n_H - f_h) // s + 1
    n_W_out = (n_W - f_w) // s + 1

    i0 = xp.repeat(xp.arange(f_h), f_w)
    i0 = xp.tile(i0, n_C)
    i1 = s * xp.repeat(xp.arange(n_H_out), n_W_out)

    j0 = xp.tile(xp.arange(f_w), f_h * n_C)
    j1 = s * xp.tile(xp.arange(n_W_out), n_H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(n_C), f_h * f_w).reshape(-1, 1)

    k = k.astype(xp.int32)
    i = i.astype(xp.int32)
    j = j.astype(xp.int32)

    cols = X[:, k, i, j]  # (m, f_h * f_w * n_C, n_H_out * n_W_out)
    cols = cols.transpose(1, 2, 0).reshape(f_h * f_w * n_C, -1)
    return cols

def _im2col_safe_batch(X, kernel_size, s):
    m = X.shape[0]
    batch = _safe_batch_size(X.shape, kernel_size, s)
    out_list = []
    for start in range(0, m, batch):
        end = min(start + batch, m)
        Xb = X[start:end]
        out_list.append(_im2col_vectorized(Xb, kernel_size, s))
    return xp.concatenate(out_list, axis=1).astype(DTYPE)

def im2col(X, kernel_size, s):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    # try to do full vectorized; caller may catch OOM and call safe batch
    try:
        return _im2col_vectorized(X, kernel_size, s)
    except Exception as ex:
        # On CuPy an OOM is raised as cupy.cuda.memory.OutOfMemoryError,
        # but catching general exceptions ensures fallback.
        return _im2col_safe_batch(X, kernel_size, s)

def _col2im_vectorized(cols, X_shape, kernel_size, s):
    """
    Channel-first col2im: X_shape = (m, C, H, W)
    cols shape (C*f*f, H_out*W_out*m)
    """
    m, C, H, W = X_shape
    f_h, f_w = kernel_size
    H_out = (H - f_h) // s + 1
    W_out = (W - f_w) // s + 1

    # Same index generation as im2col
    i0 = xp.repeat(xp.arange(f_h), f_w)
    i0 = xp.tile(i0, C)
    i1 = s * xp.repeat(xp.arange(H_out), W_out)

    j0 = xp.tile(xp.arange(f_w), f_h * C)
    j1 = s * xp.tile(xp.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)   # (C*f*f, H_out*W_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)   # (C*f*f, H_out*W_out)
    k = xp.repeat(xp.arange(C), f_h * f_w).reshape(-1, 1)  # (C*f*f, 1)

    # Prepare flat indices
    cols_reshaped = cols.reshape(C * f_h * f_w, H_out * W_out, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)  # (m, C*f*f, H_out*W_out)

    #batch_idx = xp.repeat(xp.arange(m), C * f * f * H_out * W_out)
    batch_idx = xp.repeat(xp.arange(m), i.size)
    total = batch_idx.size
    k_idx = xp.tile(k, (1, H_out * W_out)).ravel()
    k_idx = xp.tile(k, m)
    k_idx = xp.repeat(xp.arange(C), f_h * f_w * H_out * W_out)
    k_idx = xp.tile(k_idx, int(total / k_idx.size))
    i_idx = xp.tile(i, (m, 1)).ravel()
    j_idx = xp.tile(j, (m, 1)).ravel()
    k_idx = k_idx.astype(xp.int32)
    i_idx = i_idx.astype(xp.int32)
    j_idx = j_idx.astype(xp.int32)

    vals = cols_reshaped.ravel()
    flat_idx = xp.ravel_multi_index((batch_idx, k_idx, i_idx, j_idx), X_shape)

    X_flat = xp.zeros(m * C * H * W, dtype=cols.dtype)
    if xp.__name__ == 'cupy':
        scatter_add(X_flat, flat_idx, vals)
    else:
        xp.add.at(X_flat, flat_idx, vals)

    return X_flat.reshape(X_shape)

def _col2im_safe_batch(cols, X_shape, kernel_size, s):
    m, C, H, W = X_shape
    f_h, f_w = kernel_size
    batch = _safe_batch_size(X_shape, kernel_size, s)
    H_out = (H - f_h) // s + 1
    W_out = (W - f_w) // s + 1
    patches = H_out * W_out

    X = xp.zeros(X_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        X_batch = _col2im_vectorized(cols_batch, (end - start, C, H, W), kernel_size, s)
        X[start:end] = X_batch
    return X

def col2im(cols, X_shape, kernel_size, s):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    try:
        return _col2im_vectorized(cols, X_shape, kernel_size, s)
    except Exception:
        return _col2im_safe_batch(cols, X_shape, kernel_size, s)
    
def _im2col_transpose_vectorized(X, kernel_size, s, output_shape):
    """
    Transposed im2col (for Conv2DTranspose).
    NCHW convention: (m, C, H, W)

    Parameters
    ----------
    X : xp.ndarray
        Input tensor (m, C, H, W).
    f : int
        Kernel size (square).
    s : int
        Stride.
    output_shape : tuple
        (m, C, H_out, W_out) expected after Conv2DTranspose.

    Returns
    -------
    cols : xp.ndarray
        Columnized patches, shape (C * f * f, m * H_out * W_out).
    """
    m, n_C, n_H, n_W = X.shape
    _, _, n_H_out, n_W_out = output_shape
    f_h, f_w = kernel_size

    # Step 1: Upsample input by inserting zeros
    H_upsampled = (n_H - 1) * s + 1
    W_upsampled = (n_W - 1) * s + 1
    upsampled = xp.zeros((m, n_C, H_upsampled, W_upsampled), dtype=X.dtype)
    upsampled[:, :, ::s, ::s] = X

    # Step 2: Extract patches like in im2col
    i0 = xp.repeat(xp.arange(f_h), f_w)
    i0 = xp.tile(i0, n_C)
    i1 = xp.repeat(xp.arange(n_H_out), n_W_out)

    j0 = xp.tile(xp.arange(f_w), f_h * n_C)
    j1 = xp.tile(xp.arange(n_W_out), n_H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(n_C), f_h * f_w).reshape(-1, 1)

    k = k.astype(xp.int32)
    i = i.astype(xp.int32)
    j = j.astype(xp.int32)

    cols = upsampled[:, k, i, j]  # (m, f*f*C, H_out*W_out)
    cols = cols.transpose(1, 2, 0).reshape(f_h * f_w * n_C, -1)
    return cols

def _im2col_transpose_safe_batch(X, kernel_size, s, output_shape):
    m = X.shape[0]
    batch = _safe_batch_size(X.shape, kernel_size, s)
    out_list = []
    for start in range(0, m, batch):
        end = min(start + batch, m)
        Xb = X[start:end]
        out_list.append(_im2col_transpose_vectorized(Xb, kernel_size, s, output_shape))
    return xp.concatenate(out_list, axis=1).astype(DTYPE)

def im2col_transpose(X, kernel_size, s, output_shape):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    try:
        return _im2col_transpose_vectorized(X, kernel_size, s, output_shape)
    except Exception:
        return _im2col_transpose_safe_batch(X, kernel_size, s, output_shape)
    
def _col2im_transpose_vectorized(cols, X_shape, kernel_size, s, output_shape):
    """
    Channel-first col2im for Conv2DTranspose.
    X_shape = (m, C, H, W)  -- input to the deconv
    output_shape = (m, C, H_out, W_out)  -- final desired output
    cols shape = (C*f*f, H_out*W_out*m)
    """
    m, C, H, W = X_shape
    _, _, H_out, W_out = output_shape
    f_h, f_w = kernel_size

    # Expanded canvas (because stride "spreads" things out)
    H_up = (H - 1) * s + 1
    W_up = (W - 1) * s + 1

    # Same indexing trick as im2col_transpose
    i0 = xp.repeat(xp.arange(f_h), f_w)
    i0 = xp.tile(i0, C)
    i1 = xp.repeat(xp.arange(H_out), W_out)

    j0 = xp.tile(xp.arange(f_w), f_h * C)
    j1 = xp.tile(xp.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), f_h * f_w).reshape(-1, 1)

    cols_reshaped = cols.reshape(C * f_h * f_w, H_out * W_out, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)  # (m, C*f*f, H_out*W_out)

    batch_idx = xp.repeat(xp.arange(m), i.size)
    total = batch_idx.size
    k_idx = xp.repeat(xp.arange(C), f_h * f_w * H_out * W_out)
    k_idx = xp.tile(k_idx, int(total / k_idx.size))
    i_idx = xp.tile(i, (m, 1)).ravel()
    j_idx = xp.tile(j, (m, 1)).ravel()

    vals = cols_reshaped.ravel()

    # Scatter-add into upsampled canvas
    flat_idx = xp.ravel_multi_index((batch_idx, k_idx, i_idx, j_idx),
                                    (m, C, H_up + f_h - 1, W_up + f_w - 1))
    X_flat = xp.zeros((m * C * (H_up + f_h - 1) * (W_up + f_w - 1)), dtype=cols.dtype)
    if xp.__name__ == 'cupy':
        scatter_add(X_flat, flat_idx, vals)
    else:
        xp.add.at(X_flat, flat_idx, vals)

    X_up = X_flat.reshape((m, C, H_up + f_h - 1, W_up + f_w - 1))

    # Final crop to match requested output shape
    H_start = (X_up.shape[2] - H_out) // 2
    W_start = (X_up.shape[3] - W_out) // 2
    return X_up[:, :, H_start:H_start + H_out, W_start:W_start + W_out]

def _col2im_transpose_safe_batch(cols, X_shape, kernel_size, s, output_shape):
    m, C, H, W = X_shape
    _, _, H_out, W_out = output_shape
    batch = _safe_batch_size(X_shape, kernel_size, s)
    patches = H_out * W_out

    X = xp.zeros(output_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        X_batch = _col2im_transpose_vectorized(
            cols_batch, (end - start, C, H, W), kernel_size, s, (end - start, C, H_out, W_out)
        )
        X[start:end] = X_batch
    return X

def col2im_transpose(cols, X_shape, kernel_size, s, output_shape):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    try:
        return _col2im_transpose_vectorized(cols, X_shape, kernel_size, s, output_shape)
    except Exception:
        return _col2im_transpose_safe_batch(cols, X_shape, kernel_size, s, output_shape)
    
def im2col_grouped(X, kernel_size, s, groups):
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
        cols_group = im2col(X_group, kernel_size, s)
        cols_list.append(cols_group)

    # Concatenate along the first dimension (channel*k*k axis)
    cols = xp.concatenate(cols_list, axis=0)
    return cols

def col2im_grouped(cols, X_shape, kernel_size, s, groups):
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
        X_reconstructed[:, start_ch:end_ch, :, :] += col2im(cols_group, (m, group_channels, H, W), kernel_size, s)

    return X_reconstructed

def im2col_transpose_grouped(X, kernel_size, s, output_shape, groups):
    """
    Group-aware im2col for transposed convolution.

    Args:
        X (ndarray): Input tensor of shape (m, C, H, W)
        kernel_size (int or tuple): Filter size (f_h, f_w)
        s (int): Stride of transposed convolution
        output_shape (tuple): Target output shape (m, C, H_out, W_out)
        groups (int): Number of groups (C must be divisible by groups)

    Returns:
        ndarray: Column-form tensor with grouped layout
                 shape (C/group * f_h * f_w * groups, H_out * W_out * m)
    """
    m, C, H, W = X.shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    group_channels = C // groups
    cols_list = []

    for g in range(groups):
        start = g * group_channels
        end = (g + 1) * group_channels
        X_group = X[:, start:end, :, :]

        # Compute per-group output shape
        out_group_shape = (
            output_shape[0], group_channels,
            output_shape[2], output_shape[3]
        )

        cols_group = _im2col_transpose_vectorized(X_group, kernel_size, s, out_group_shape)
        cols_list.append(cols_group)

    cols = xp.concatenate(cols_list, axis=0)
    return cols

def col2im_transpose_grouped(cols, X_shape, kernel_size, s, output_shape, groups):
    """
    Group-aware col2im for transposed convolution.

    Args:
        cols (ndarray): Column tensor from grouped transposed convolution
        X_shape (tuple): Original input shape (m, C, H, W)
        kernel_size (int or tuple): Filter size (f_h, f_w)
        s (int): Stride of transposed convolution
        output_shape (tuple): Target output shape (m, C, H_out, W_out)
        groups (int): Number of groups (C must be divisible by groups)

    Returns:
        ndarray: Reconstructed 4D tensor (m, C, H_out, W_out)
    """
    m, C, H, W = X_shape
    if C % groups != 0:
        raise ValueError(f"Number of channels ({C}) must be divisible by groups ({groups})")

    group_channels = C // groups
    cols_per_group = cols.shape[0] // groups

    X_reconstructed = xp.zeros(output_shape, dtype=cols.dtype)

    for g in range(groups):
        start_cols = g * cols_per_group
        end_cols = (g + 1) * cols_per_group
        start_ch = g * group_channels
        end_ch = (g + 1) * group_channels

        cols_group = cols[start_cols:end_cols, :]
        group_output_shape = (
            output_shape[0], group_channels,
            output_shape[2], output_shape[3]
        )

        X_reconstructed[:, start_ch:end_ch, :, :] += _col2im_transpose_vectorized(
            cols_group,
            (m, group_channels, H, W),
            kernel_size,
            s,
            group_output_shape
        )

    return X_reconstructed

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