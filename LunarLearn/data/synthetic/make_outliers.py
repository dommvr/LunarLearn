import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng

xp = backend.xp
DTYPE = backend.DTYPE


def make_outliers(
    n_samples=2000,
    n_features=2,
    outlier_ratio=0.05,          # fraction of points that are outliers
    inlier_std=1.0,
    outlier_mode="uniform",      # "uniform" or "far_gaussian"
    box_scale=8.0,               # for uniform: outliers in [-box_scale, box_scale]^d
    far_center_scale=8.0,        # for far_gaussian: center ~ N(0, far_center_scale^2)
    outlier_std=0.5,             # for far_gaussian
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Anomaly dataset: one main Gaussian cluster + a small set of outliers.
    Returns:
        X: (n_samples, n_features)
        y: (n_samples,) with 0=inlier, 1=outlier
    """
    if dtype is None:
        dtype = DTYPE
    if outlier_ratio < 0 or outlier_ratio >= 1:
        raise ValueError("outlier_ratio must be in [0, 1)")
    if n_features < 1:
        raise ValueError("n_features must be >= 1")

    rng = _get_rng(random_state)

    n_out = int(round(float(outlier_ratio) * n_samples))
    n_in = n_samples - n_out

    # Inliers
    Xin = rng.normal(0.0, 1.0, size=(n_in, n_features)).astype(dtype)
    Xin = Xin * xp.asarray(float(inlier_std), dtype=dtype)
    yin = xp.zeros((n_in,), dtype=xp.int64)

    # Outliers
    if n_out > 0:
        if outlier_mode == "uniform":
            lo = xp.full((n_features,), -float(box_scale), dtype=dtype)
            hi = xp.full((n_features,), float(box_scale), dtype=dtype)
            Xo = rng.uniform(0.0, 1.0, size=(n_out, n_features)).astype(dtype)
            Xo = lo + (hi - lo) * Xo
        elif outlier_mode == "far_gaussian":
            # Place a far-away Gaussian cluster
            center = rng.normal(0.0, float(far_center_scale), size=(n_features,)).astype(dtype)
            Xo = rng.normal(0.0, 1.0, size=(n_out, n_features)).astype(dtype)
            Xo = Xo * xp.asarray(float(outlier_std), dtype=dtype) + center
        else:
            raise ValueError('outlier_mode must be "uniform" or "far_gaussian"')

        yo = xp.ones((n_out,), dtype=xp.int64)

        X = xp.concatenate([Xin, Xo], axis=0)
        y = xp.concatenate([yin, yo], axis=0)
    else:
        X, y = Xin, yin

    if shuffle:
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y