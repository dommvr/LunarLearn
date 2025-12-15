import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng

xp = backend.xp
DTYPE = backend.DTYPE


def make_gaussian_quantiles(
    n_samples=2000,
    n_features=2,
    n_classes=3,
    noise=0.0,
    elliptical=False,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Gaussian quantiles dataset.
    Sample X ~ N(0, I), compute squared radius s = ||X||^2,
    then bucket s by quantiles into n_classes rings/shells.

    Returns:
        X: (n_samples, n_features), y: (n_samples,)
    """
    if dtype is None:
        dtype = DTYPE
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")

    rng = _get_rng(random_state)

    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(dtype)

    # Optional: make it elliptical via a random linear transform
    if elliptical:
        A = rng.normal(0.0, 1.0, size=(n_features, n_features)).astype(dtype)
        # Make it a bit better-conditioned by nudging diagonal
        A = A + (0.5 * xp.eye(n_features, dtype=dtype))
        X = X @ A

    if noise and noise > 0:
        X = X + rng.normal(0.0, noise, size=X.shape).astype(dtype)

    s = xp.sum(X * X, axis=1)  # squared radius, shape (n_samples,)

    # Compute thresholds at quantiles without relying on xp.quantile quirks:
    s_sorted = xp.sort(s)
    # thresholds for k=1..n_classes-1 at k/n_classes
    thresholds = []
    for k in range(1, n_classes):
        q = k / n_classes
        pos = int(q * (n_samples - 1))
        thresholds.append(s_sorted[pos])

    thresholds = xp.asarray(thresholds, dtype=dtype)

    # Digitize manually: count how many thresholds s exceeds
    # y = number of thresholds where s > thr
    y = xp.zeros((n_samples,), dtype=xp.int64)
    for thr in thresholds:
        y = y + (s > thr).astype(xp.int64)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y