import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _allocate_counts

xp = backend.xp
DTYPE = backend.DTYPE


def make_varying_density_blobs(
    n_samples=2000,
    centers=None,                 # array-like (n_clusters, n_features). If None, default 3 centers in 2D.
    cluster_stds=(0.2, 0.8, 1.6),  # per-cluster std (vary density)
    weights=None,                 # per-cluster sample weights, defaults uniform
    noise_ratio=0.0,              # fraction of extra uniform noise points
    noise_box=None,               # (low, high) or (low_vec, high_vec)
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Clustering stress-test: clusters with different variances (densities) and optional noise points.
    Returns:
        X: (n_samples + n_noise, n_features)
        y: (n_samples + n_noise,) where noise points have label -1
    """
    if dtype is None:
        dtype = DTYPE

    rng = _get_rng(random_state)

    # Default: 3 clusters in 2D with similar spacing
    if centers is None:
        centers = xp.asarray([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]], dtype=dtype)
    else:
        centers = xp.asarray(centers, dtype=dtype)

    n_clusters = int(centers.shape[0])
    n_features = int(centers.shape[1])

    if len(cluster_stds) != n_clusters:
        raise ValueError(f"cluster_stds must have length {n_clusters}, got {len(cluster_stds)}")

    if weights is None:
        weights = [1.0 / n_clusters] * n_clusters
    else:
        if len(weights) != n_clusters:
            raise ValueError(f"weights must have length {n_clusters}, got {len(weights)}")
        s = float(sum(weights))
        if s <= 0:
            raise ValueError("weights must sum to a positive number")
        weights = [w / s for w in weights]

    # Noise points count (additive)
    if noise_ratio < 0:
        raise ValueError("noise_ratio must be >= 0")
    n_noise = int(round(float(noise_ratio) * n_samples))

    # Main blob counts
    counts = _allocate_counts(n_samples, weights)

    X_parts = []
    y_parts = []

    for k in range(n_clusters):
        nk = counts[k]
        if nk <= 0:
            continue
        std = float(cluster_stds[k])
        Xk = rng.normal(0.0, 1.0, size=(nk, n_features)).astype(dtype)
        Xk = Xk * xp.asarray(std, dtype=dtype) + centers[k]  # broadcast center
        X_parts.append(Xk)
        y_parts.append(xp.full((nk,), k, dtype=xp.int64))

    X = xp.concatenate(X_parts, axis=0) if X_parts else xp.empty((0, n_features), dtype=dtype)
    y = xp.concatenate(y_parts, axis=0) if y_parts else xp.empty((0,), dtype=xp.int64)

    # Uniform noise points (label -1)
    if n_noise > 0:
        if noise_box is None:
            # Default noise box based on centers extent
            lo = xp.min(centers, axis=0) - xp.asarray(3.0, dtype=dtype)
            hi = xp.max(centers, axis=0) + xp.asarray(3.0, dtype=dtype)
        else:
            lo, hi = noise_box
            lo = xp.asarray(lo, dtype=dtype)
            hi = xp.asarray(hi, dtype=dtype)
            if lo.shape == ():
                lo = xp.full((n_features,), lo, dtype=dtype)
            if hi.shape == ():
                hi = xp.full((n_features,), hi, dtype=dtype)

        Xn = rng.uniform(0.0, 1.0, size=(n_noise, n_features)).astype(dtype)
        Xn = lo + (hi - lo) * Xn
        yn = xp.full((n_noise,), -1, dtype=xp.int64)

        X = xp.concatenate([X, Xn], axis=0)
        y = xp.concatenate([y, yn], axis=0)

    if shuffle:
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y