import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng

xp = backend.xp
DTYPE = backend.DTYPE


def make_regression(
    n_samples=100,
    n_features=20,
    n_informative=10,
    n_targets=1,
    bias=0.0,
    noise=0.0,
    correlation=0.0,         # 0.. <1, Toeplitz cov: rho^|i-j|
    heteroscedastic=0.0,     # 0.., noise grows with |signal|
    shuffle=True,
    coef=False,
    random_state=None,
    dtype=None,
):
    """
    Rich regression generator with optional correlated features and heteroscedastic noise.

    Returns:
        X: (n_samples, n_features)
        y: (n_samples,) or (n_samples, n_targets)
        (optional) w: true coefficients in feature space, shape (n_features, n_targets)
    """
    if dtype is None:
        dtype = DTYPE

    if n_informative > n_features:
        raise ValueError("n_informative must be <= n_features")
    if n_targets < 1:
        raise ValueError("n_targets must be >= 1")
    if correlation < 0 or correlation >= 1:
        raise ValueError("correlation must be in [0, 1)")

    rng = _get_rng(xp, random_state)

    # Base Gaussian
    Z = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(dtype)

    # Correlate features via Toeplitz covariance if needed
    if correlation > 0:
        idx = xp.arange(n_features)
        cov = correlation ** xp.abs(idx[:, None] - idx[None, :])
        cov = cov.astype(dtype)
        L = xp.linalg.cholesky(cov + (1e-6 * xp.eye(n_features, dtype=dtype)))
        X = Z @ L.T
    else:
        X = Z

    # True coefficients: only informative features matter
    w_inf = rng.normal(0.0, 1.0, size=(n_informative, n_targets)).astype(dtype)
    w = xp.zeros((n_features, n_targets), dtype=dtype)
    w[:n_informative] = w_inf

    signal = (X[:, :n_informative] @ w_inf) + bias  # (n_samples, n_targets)

    y = signal

    # Add noise (optionally heteroscedastic)
    if noise and noise > 0:
        if heteroscedastic and heteroscedastic > 0:
            scale = 1.0 + heteroscedastic * xp.abs(signal)
            eps = rng.normal(0.0, 1.0, size=signal.shape).astype(dtype) * (noise * scale)
        else:
            eps = rng.normal(0.0, noise, size=signal.shape).astype(dtype)
        y = y + eps

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

        feat_perm = rng.permutation(n_features)
        X = X[:, feat_perm]
        w = w[feat_perm]

    # Squeeze y for single target
    if n_targets == 1:
        y = y.reshape(-1)

    if coef:
        return X.astype(dtype), y, w.astype(dtype)
    return X.astype(dtype), y