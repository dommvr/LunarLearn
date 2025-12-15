import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_spirals(
    n_samples=2000,
    n_classes=2,
    turns=3.0,
    noise=0.2,
    radius_scale=1.0,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Spiral dataset (2D), for 2 or 3 classes.
    Returns:
        X: (n_samples, 2), y: (n_samples,)
    """
    if dtype is None:
        dtype = DTYPE
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if turns <= 0:
        raise ValueError("turns must be > 0")

    rng = _get_rng(random_state)

    # Distribute samples across classes
    base = n_samples // n_classes
    rem = n_samples % n_classes
    counts = [base + (1 if c < rem else 0) for c in range(n_classes)]

    X_list = []
    y_list = []

    max_t = float(turns) * 2.0 * math.pi

    for c, n_c in enumerate(counts):
        if n_c == 0:
            continue

        # Parameter t along spiral
        t = rng.uniform(0.0, max_t, size=(n_c,)).astype(dtype)

        # Radius grows with t; scale makes it nicer
        r = radius_scale * (t / max_t)  # in [0, radius_scale]

        # Angle with class offset
        theta = t + (2.0 * math.pi * c / n_classes)

        x0 = r * xp.cos(theta)
        x1 = r * xp.sin(theta)
        Xc = xp.stack([x0, x1], axis=1).astype(dtype)

        if noise and noise > 0:
            Xc = Xc + rng.normal(0.0, noise, size=Xc.shape).astype(dtype)

        X_list.append(Xc)
        y_list.append(xp.full((n_c,), c, dtype=xp.int64))

    X = xp.concatenate(X_list, axis=0) if X_list else xp.empty((0, 2), dtype=dtype)
    y = xp.concatenate(y_list, axis=0) if y_list else xp.empty((0,), dtype=xp.int64)

    if shuffle:
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y