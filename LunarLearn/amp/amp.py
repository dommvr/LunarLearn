from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Callable, Iterable

import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor
from LunarLearn.amp import DynamicLossScaler

xp = backend.xp

class _AmpState:
    @property
    def available(self) -> bool:
        return backend.MIXED_PRECISION and backend.gpu_available()
    
    enabled: bool = False
    active: bool = False
    compute_dtype = backend.C_DTYPE  # fp16
    promote_dtype = backend.DTYPE    # fp32
    scaler = DynamicLossScaler(init_scale=backend.SCALING_FACTOR)

    # Default op policy
    FP16_SAFE = {
        "abs", "neg", "tanh", "relu", "nll_loss", "clip",
        "add", "subtract", "multiply", "maximum",
        "minimum", "where",
        "matmul", "dot", "einsum",
        "transpose", "reshape", "squeeze", "unsqueeze",
        "concatenate", "stack", "split", "slice",
        "flip", "roll", "gather", "scatter",
        "expand", "repeat", "tile", "pad",
        "im2col", "col2im", "im2col_transpose", "col2im_transpose",
        "im2col_grouped", "col2im_grouped", 
        "im2col_transpose_grouped", "col2im_transpose_grouped",
        "dropout", "upsample", "avg_pool2d", "sort", "cumsum",
        "multinomial", "searchsorted"
    }
    FP32_FORCE = {
        "zeros", "ones", "full", "arange", "eye",
        "exp", "log", "sqrt", "erf", "erfc",
        "divide", "power",
        "eq", "ne", "lt", "le", "gt", "ge",
        "logical_add", "logical_or", "logical_xor",
        "logical_not",
        "max", "min", "argmax", "argmin",
        "inv", "det", "trace", "svd",
        "normalize", "normalize_absmax", "renorm"
    }
    REDUCE_FP32 = {"sum", "mean", "var", "std", "logsumexp", "norm",
                   "sigmoid", "softmax", "log_softmax", "cross_entropy"}

STATE = _AmpState()

def is_available() -> bool:
    return STATE.available

@contextmanager
def autocast(enabled: bool = True, dtype: Any | None = None):
    """Enable/disable AMP inside a `with` block."""
    STATE.enabled = bool(enabled) if STATE.available else False
    STATE.active = bool(enabled) if STATE.available else False
    old_dtype = STATE.compute_dtype
    if dtype is not None:
        STATE.compute_dtype = dtype
    try:
        yield
    finally:
        STATE.enabled = False
        STATE.compute_dtype = old_dtype

def _promote_dtype(args):
    for arg in args:
        if hasattr(arg, "dtype") and arg.dtype == backend.DTYPE:
            return backend.DTYPE
    return backend.C_DTYPE

def _cast_arg_to(a: Any, dtype) -> Any:
    if isinstance(a, Tensor) and a.data.dtype != dtype:
        # Respect global grad flag through your astype op.
        return a.astype(dtype, copy=False)
    return a

def _walk_and_cast(args, dtype):
    if isinstance(args, (list, tuple)):
        return type(args)(_walk_and_cast(x, dtype) for x in args)
    return _cast_arg_to(args, dtype)

def _want_fp32(opname: str) -> bool:
    return (opname in STATE.FP32_FORCE) or (opname in STATE.REDUCE_FP32)

def _want_fp16(opname: str) -> bool:
    return opname in STATE.FP16_SAFE

def _maybe_promote_output(opname: str, out: Tensor, in_dtype=None) -> Tensor:
    """
    For reduce_fp32 ops: compute in fp32 but cast back to input dtype if safe.

    Args:
        opname (str): Name of the op.
        out (Tensor): Output tensor after computation.
        in_dtype (dtype): Original input dtype before promotion.
    """
    if opname in STATE.REDUCE_FP32 and isinstance(out, Tensor):
        # If AMP is enabled and original input was fp16, cast back
        if in_dtype == backend.C_DTYPE:
            return out.astype(backend.C_DTYPE, copy=False)
        # Otherwise, stay in fp32
        return out.astype(backend.DTYPE, copy=False)
    return out

def dispatch_amp(opname: str, fn: Callable, *args, **kwargs):
    """
    Route op execution through AMP policy. 
    - If AMP disabled: call fn as-is.
    - If FP32 op: cast float tensors to fp32.
    - If FP16-safe op: cast float tensors to fp16 (compute_dtype).
    """
    if not STATE.enabled or not STATE.available:
        return fn(*args, **kwargs)

    in_dtype = _promote_dtype(args)

    # Decide compute dtype
    if _want_fp32(opname):
        compute_dtype = backend.DTYPE
    elif _want_fp16(opname):
        compute_dtype = STATE.compute_dtype
    else:
        # Unknown op -> conservative default: fp32
        compute_dtype = backend.DTYPE

    # Cast inputs
    cargs = _walk_and_cast(args, compute_dtype)

    # Compute
    out = fn(*cargs, **kwargs)

    # Post-process output for reduce ops (keep in fp32)
    out = _maybe_promote_output(opname, out, in_dtype)
    return out

def scale_loss(loss):
    if not STATE.active:
        return loss
    return STATE.scaler.scale_loss(loss)

def unscale_grads(model):
    if not STATE.active:
        return True
    return STATE.scaler.unscale_grads(model)

def step_if_ready(optimizer, model):
    if unscale_grads(model):
        optimizer.step(model.parameters(with_layer=True))