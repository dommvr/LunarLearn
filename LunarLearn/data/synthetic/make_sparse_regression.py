import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _sample_values

xp = backend.xp
DTYPE = backend.DTYPE


def make_sparse_regression(
    n_samples=2000,
    n_features=30000,
    n_informative=100,
    density=0.001,
    noise=0.0,
    bias=0.0,
    distribution="lognormal",   # poisson/exponential/lognormal/normal
    nonnegative=False,          # allow signed features by default for regression
    tfidf_like=False,           # regression often prefers raw-ish features
    coef=False,                 # return true weights
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    High-dimensional sparse regression generator (dense matrix with many zeros).
    Returns:
        X: (n_samples, n_features)
        y: (n_samples,)
        (optional) w: (n_features,)
        informative_idx: indices of informative features
    """
    if dtype is None:
        dtype = DTYPE
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if n_informative < 1 or n_informative > n_features:
        raise ValueError("n_informative must be in [1, n_features]")
    if density <= 0 or density > 1:
        raise ValueError("density must be in (0, 1]")

    rng = _get_rng(random_state)

    informative_idx = rng.choice(n_features, size=(n_informative,), replace=False).astype(xp.int64)

    # sparse true coefficients
    w = xp.zeros((n_features,), dtype=dtype)
    w_inf = rng.normal(0.0, 1.0, size=(n_informative,)).astype(dtype)
    w[informative_idx] = w_inf

    X = xp.zeros((n_samples, n_features), dtype=dtype)

    mean_k = float(density) * float(n_features)

    for i in range(n_samples):
        k = int(rng.poisson(lam=max(mean_k, 1.0)))
        if k < 1:
            k = 1
        if k > n_features:
            k = n_features

        cols = rng.choice(n_features, size=(k,), replace=False).astype(xp.int64)
        vals = _sample_values(rng, k, distribution, dtype, nonnegative=nonnegative)

        X[i, cols] = vals

    if tfidf_like:
        X = xp.log1p(xp.maximum(X, xp.asarray(0, dtype=dtype)))
        denom = xp.sqrt(xp.sum(X * X, axis=1, keepdims=True)) + xp.asarray(1e-12, dtype=dtype)
        X = X / denom

    y = X @ w + xp.asarray(bias, dtype=dtype)

    if noise and noise > 0:
        y = y + rng.normal(0.0, noise, size=y.shape).astype(dtype)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    if coef:
        return X.astype(dtype), y.astype(dtype), w.astype(dtype), informative_idx
    return X.astype(dtype), y.astype(dtype), informative_idx