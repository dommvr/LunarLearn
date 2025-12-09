import LunarLearn.core.backend.backend as backend


xp = backend.xp

if xp.__name__ == "cupy":
    try:
        from cupyx.scatter_add import scatter_add as _scatter_add
    except ImportError:
        from cupyx._scatter import scatter_add as _scatter_add

    def row_scatter_add(arr, idx, updates):
        # arr: (N, k) or (N,)
        if arr.ndim == 1:
            _scatter_add(arr, (idx,), updates)
        elif arr.ndim == 2:
            _scatter_add(arr, (idx, slice(None)), updates)
        else:
            raise ValueError("row_scatter_add only supports 1D or 2D arrays.")
else:
    def row_scatter_add(arr, idx, updates):
        xp.add.at(arr, idx, updates)