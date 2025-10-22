"""
Backend runtime selector for LunarLearn.

- Single import point for array backend (`xp`) and core runtime flags.
- Seamlessly toggle CPU (NumPy) / GPU (CuPy).
- Centralized dtype, mixed precision, safety/memory helpers.
- Minimal API surface with global-access pattern:
    >>> import LunarLearn.backend as backend
    >>> xp = backend.xp
    >>> DTYPE = backend.DTYPE

This module is intentionally stateful to be easy to use in userland code.
"""

from __future__ import annotations

import os
import time
import numpy as _np
from contextlib import contextmanager
from LunarLearn.backend.config import CONFIG


# ---------------------------
# Optional GPU backend (CuPy)
# ---------------------------
try:
    import cupy as _cp
    _CUPY_AVAILABLE = True
except Exception:
    _cp = None
    _CUPY_AVAILABLE = False


# ---------------------------
# Public runtime state (globals)
# ---------------------------
xp = _np                       # current array module (NumPy or CuPy)
USING = "cpu"                  # "cpu" | "gpu"
SEED = CONFIG.get("seed", 997)

# Dtypes
DTYPE = _np.float32            # master/FP32
C_DTYPE = _np.float16          # compute/FP16 for mixed precision (cast-at-edges)
GLOBAL_DTYPE = _np.float32     # default dtype for new tensors

# Safety / memory hints
SAFE_FACTOR = CONFIG.get("safe_factor", 0.8)              # fraction of free GPU memory to consider "available" for big ops

# Mixed precision & scaling
MIXED_PRECISION = CONFIG.get("mixed_precision", True)
SCALING_FACTOR = CONFIG.get("scaling_factor", 1024)

# Autograd switch
AUTOGRAD_ENABLED = CONFIG.get("autograd_enable", True)


# ===========================
# Introspection / utilities
# ===========================
def gpu_available() -> bool:
    """Return True if CuPy is importable and at least one GPU is accessible."""
    return _CUPY_AVAILABLE


def is_gpu() -> bool:
    """Return True if current backend is GPU (CuPy)."""
    return USING == "gpu"


def device_name() -> str:
    """Human-readable device name."""
    if is_gpu() and _cp is not None:
        try:
            dev_id = _cp.cuda.Device().id
            props = _cp.cuda.runtime.getDeviceProperties(dev_id)
            name = props.get("name", b"GPU").decode(errors="ignore")
            return f"GPU:{dev_id} ({name})"
        except Exception:
            return "GPU (CuPy)"
    return "CPU (NumPy)"


def get_device() -> str:
    """Return current device string: 'cpu' or 'gpu'."""
    return USING


def memory_info():
    """
    Return a tuple (free_bytes, total_bytes) if available.

    - On GPU: uses CuPy runtime memGetInfo.
    - On CPU: returns (None, None) as a placeholder (no psutil dependency).
    """
    if is_gpu() and _cp is not None:
        try:
            free_b, total_b = _cp.cuda.runtime.memGetInfo()
            return int(free_b), int(total_b)
        except Exception:
            return None, None
    return None, None


def synchronize():
    """Block until all queued ops on the current device are complete."""
    if is_gpu() and _cp is not None:
        try:
            _cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


# ===========================
# Backend switching
# ===========================
def _set_globals_for_numpy():
    global xp, USING, DTYPE, C_DTYPE, GLOBAL_DTYPE
    xp = _np
    USING = "cpu"
    # keep dtypes as NumPy types
    if DTYPE is None or isinstance(DTYPE, str):
        DTYPE = _np.float32
    if C_DTYPE is None or isinstance(C_DTYPE, str):
        C_DTYPE = _np.float16
    if GLOBAL_DTYPE is None or isinstance(GLOBAL_DTYPE, str):
        GLOBAL_DTYPE = _np.float32


def _set_globals_for_cupy():
    global xp, USING, DTYPE, C_DTYPE, GLOBAL_DTYPE
    xp = _cp
    USING = "gpu"
    # mirror dtype to CuPy (types are compatible across backends)
    DTYPE = _cp.float32 if DTYPE == _np.float32 else _cp.float64
    C_DTYPE = _cp.float16
    GLOBAL_DTYPE = _cp.float32 if GLOBAL_DTYPE == _np.float32 else _cp.float64


# Device auto-select from config
def _auto_select_device():
    global xp, USING
    device = CONFIG.get("device", "cpu").lower()
    if device == "gpu" and _CUPY_AVAILABLE:
        use_gpu()
    else:
        use_cpu()


def _set_default_dtype():
    global GLOBAL_DTYPE, DTYPE
    dtype_str = CONFIG.get("dtype", "float32")
    dtype_map = {"float16": _np.float16, "float32": _np.float32, "float64": _np.float64}
    GLOBAL_DTYPE = dtype_map[dtype_str]
    DTYPE = GLOBAL_DTYPE


def warmup_cupy(iterations: int = 3, size: int = 512):
    """
    Warm up CUDA context & JIT caches for fair timing and consistent first-run performance.
    """
    if not (is_gpu() and _cp is not None):
        return
    print("CuPy warm-up...")
    a = _cp.random.randn(size, size, dtype=_cp.float32)
    b = _cp.random.randn(size, size, dtype=_cp.float32)
    for _ in range(iterations):
        _ = a @ b
        _ = a * b
    synchronize()
    print("CuPy warm-up done.")


def use_gpu(warmup: bool = True):
    """
    Switch backend to GPU (CuPy).
    Raises ImportError if CuPy is not available.
    """
    if not _CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Run `pip install cupy` to use GPU.")
    _set_globals_for_cupy()
    # ensure reproducibility on GPU too
    try:
        _cp.random.seed(SEED)
    except Exception:
        pass
    print(f"Using {device_name()}")
    if warmup:
        warmup_cupy()


def use_cpu():
    """Switch backend to CPU (NumPy)."""
    _set_globals_for_numpy()
    _np.random.seed(SEED)
    print(f"Using {device_name()}")


# Initialize to CPU by default
_set_default_dtype()
_set_globals_for_numpy()
_np.random.seed(SEED)

# Select device from config
_auto_select_device()


# ===========================
# Runtime configuration
# ===========================
def set_seed(seed: int):
    """Set RNG seed for both NumPy and CuPy (if present)."""
    global SEED
    SEED = int(seed)
    _np.random.seed(SEED)
    if _CUPY_AVAILABLE:
        try:
            _cp.random.seed(SEED)
        except Exception:
            pass


def set_float(float_: str = "float32"):
    """
    Backward-compat alias for dtype selection.
    Prefer `set_dtype("float32" | "float64")`.
    """
    set_dtype(float_)


def set_dtype(dtype: str = "float32"):
    """
    Set master DTYPE and GLOBAL_DTYPE to float32 or float64.
    """
    global DTYPE, GLOBAL_DTYPE
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    if is_gpu() and _cp is not None:
        DTYPE = _cp.float32 if dtype == "float32" else _cp.float64
        GLOBAL_DTYPE = DTYPE
    else:
        DTYPE = _np.float32 if dtype == "float32" else _np.float64
        GLOBAL_DTYPE = DTYPE


def set_compute_float(dtype: str = "float16"):
    """
    Set compute dtype (C_DTYPE) used for mixed precision casting at compute boundaries.
    Typically 'float16'. (BF16 can be added later.)
    """
    global C_DTYPE
    if dtype not in ("float16",):
        raise ValueError("Only 'float16' is supported currently.")
    if is_gpu() and _cp is not None:
        C_DTYPE = _cp.float16
    else:
        C_DTYPE = _np.float16


def set_mixed_precision(enabled: bool = True):
    """Enable/disable mixed precision casting."""
    global MIXED_PRECISION
    MIXED_PRECISION = bool(enabled)


def set_scaling_factor(scale: float = 1024):
    """Set loss scaling factor used by dynamic/static loss scalers."""
    global SCALING_FACTOR
    SCALING_FACTOR = float(scale)


def set_safety_factor(factor: float = 0.8):
    """Set fraction of free GPU memory treated as 'safe' for large ops."""
    global SAFE_FACTOR
    SAFE_FACTOR = float(factor)


# ===========================
# Autograd guards
# ===========================
def is_grad_enabled() -> bool:
    """Return whether autograd recording is enabled."""
    return AUTOGRAD_ENABLED


@contextmanager
def no_grad():
    global AUTOGRAD_ENABLED
    _prev = AUTOGRAD_ENABLED
    AUTOGRAD_ENABLED = False
    try:
        yield
    finally:
        AUTOGRAD_ENABLED = _prev


@contextmanager
def enabled_grad():
    global AUTOGRAD_ENABLED
    _prev = AUTOGRAD_ENABLED
    AUTOGRAD_ENABLED = True
    try:
        yield
    finally:
        AUTOGRAD_ENABLED = _prev