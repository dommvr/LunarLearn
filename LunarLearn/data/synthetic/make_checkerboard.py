import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _flip_labels

xp = backend.xp
DTYPE = backend.DTYPE


def make_checkerboard(
    n_samples=5000,
    n_squares=4,
    low=0.0,
    high=1.0,
    feature_noise=0.0,
    flip_y=0.0,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Checkerboard dataset in 2D.
    Partition the plane into n_squares x n_squares cells, alternate labels by parity.
    Returns:
        X: (n_samples, 2), y: (n_samples,)
    """
    if dtype is None:
        dtype = DTYPE
    if n_squares < 1:
        raise ValueError("n_squares must be >= 1")
    if not (high > low):
        raise ValueError("high must be > low")

    rng = _get_rng(random_state)

    X = rng.uniform(low, high, size=(n_samples, 2)).astype(dtype)

    if feature_noise and feature_noise > 0:
        X = X + rng.normal(0.0, feature_noise, size=X.shape).astype(dtype)

    # Map to [0, 1) for indexing
    Xn = (X - low) / (high - low)
    Xn = xp.clip(Xn, 0.0, 1.0 - xp.finfo(dtype).eps)

    i = xp.floor(Xn[:, 0] * n_squares).astype(xp.int64)
    j = xp.floor(Xn[:, 1] * n_squares).astype(xp.int64)

    y = ((i + j) % 2).astype(xp.int64)
    y = _flip_labels(y, n_classes=2, flip_y=flip_y, rng=rng)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y