import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _flip_labels

xp = backend.xp
DTYPE = backend.DTYPE


def make_xor(
    n_samples=1000,
    low=-1.0,
    high=1.0,
    feature_noise=0.0,
    flip_y=0.0,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    XOR dataset in 2D.
    Label rule: (x0 > 0) XOR (x1 > 0)
    Returns:
        X: (n_samples, 2), y: (n_samples,)
    """
    if dtype is None:
        dtype = DTYPE

    rng = _get_rng(random_state)

    X = rng.uniform(low, high, size=(n_samples, 2)).astype(dtype)

    if feature_noise and feature_noise > 0:
        X = X + rng.normal(0.0, feature_noise, size=X.shape).astype(dtype)

    a = X[:, 0] > 0
    b = X[:, 1] > 0
    y = xp.logical_xor(a, b).astype(xp.int64)

    y = _flip_labels(y, n_classes=2, flip_y=flip_y, rng=rng)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y