class gpu_scope:
    """Context manager to temporarily switch computations to GPU."""
    def __enter__(self):
        import LunarLearn.core.backend.backend as backend
        self.prev_xp = backend.xp
        self.prev_using = backend.USING
        
        if not backend.gpu_available():
            raise RuntimeError("GPU not available.")
        backend.use_gpu()
        return backend.xp  # optional: lets user grab xp if needed

    def __exit__(self, exc_type, exc_value, tb):
        import LunarLearn.core.backend.backend as backend
        backend.xp = self.prev_xp
        backend.USING = self.prev_using
        if backend.USING == "gpu":
            backend.xp.cuda.Device().synchronize()

class mixed_precision:
    """Temporarily enable or disable mixed precision inside a block."""
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        import LunarLearn.core.backend.backend as backend
        self.prev_mp = backend.MIXED_PRECISION
        backend.MIXED_PRECISION = self.enabled
        return backend.MIXED_PRECISION

    def __exit__(self, exc_type, exc_value, tb):
        import LunarLearn.core.backend.backend as backend
        backend.MIXED_PRECISION = self.prev_mp


class precision_scope:
    """
    Temporarily change the global floating-point precision (dtype) inside a `with` block.
    
    This affects:
      - Tensor creation (`xp.zeros`, `xp.random.randn`, etc.)
      - Layer initialization defaults
      - Weight casting in forward passes (if layers respect GLOBAL_DTYPE)
    
    Args:
        dtype (str or dtype): Precision to use ("float16", "float32", "float64", xp.float16, etc.)
    """
    def __init__(self, dtype="float32"):
        import LunarLearn.core.backend.backend as backend
        # Support both string and actual dtype
        if isinstance(dtype, str):
            dtype_map = {
                "float16": backend.xp.float16,
                "float32": backend.xp.float32,
                "float64": backend.xp.float64,
            }
            if dtype not in dtype_map:
                raise ValueError(f"Unsupported dtype '{dtype}'. Use one of: {list(dtype_map.keys())}")
            self.new_dtype = dtype_map[dtype]
        else:
            self.new_dtype = dtype

    def __enter__(self):
        import LunarLearn.core.backend.backend as backend
        self.prev_dtype = backend.GLOBAL_DTYPE
        backend.GLOBAL_DTYPE = self.new_dtype
        return backend.GLOBAL_DTYPE

    def __exit__(self, exc_type, exc_value, tb):
        import LunarLearn.core.backend.backend as backend
        backend.GLOBAL_DTYPE = self.prev_dtype


class precision_and_device:
    """
    Combined context manager to temporarily set both the computation device 
    (CPU or GPU) and global precision (dtype).

    Args:
        device (str): "cpu" or "gpu". Default: current device.
        dtype (str or dtype): Precision type ("float16", "float32", "float64").
                              Can also accept backend.xp.float32 etc.
    """
    def __init__(self, device=None, dtype=None):
        import LunarLearn.core.backend.backend as backend
        self.device = device or backend.USING
        self.dtype = dtype or backend.GLOBAL_DTYPE

    def __enter__(self):
        import LunarLearn.core.backend.backend as backend

        # Save previous state
        self.prev_device = backend.USING
        self.prev_dtype = backend.GLOBAL_DTYPE

        # Switch device
        if self.device.lower() == "gpu" and backend.CUPY_AVAILABLE:
            backend.use_gpu()
        elif self.device.lower() == "cpu":
            backend.use_cpu()
        else:
            raise ValueError(
                f"Invalid device '{self.device}'. Must be 'cpu' or 'gpu' (GPU available: {backend.CUPY_AVAILABLE})"
            )

        # Switch precision
        if isinstance(self.dtype, str):
            dtype_map = {
                "float16": backend.xp.float16,
                "float32": backend.xp.float32,
                "float64": backend.xp.float64,
            }
            backend.GLOBAL_DTYPE = dtype_map[self.dtype]
        else:
            backend.GLOBAL_DTYPE = self.dtype

        return backend.xp, backend.GLOBAL_DTYPE

    def __exit__(self, exc_type, exc_value, tb):
        import LunarLearn.core.backend.backend as backend

        # Restore device and dtype
        if self.prev_device == "gpu":
            backend.use_gpu()
        else:
            backend.use_cpu()
        backend.GLOBAL_DTYPE = self.prev_dtype