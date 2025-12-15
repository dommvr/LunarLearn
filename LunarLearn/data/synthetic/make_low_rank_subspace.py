import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng

xp = backend.xp
DTYPE = backend.DTYPE


def make_low_rank_subspace(
    n_samples=2000,
    n_features=50,       # ambient dimension d
    rank=5,              # intrinsic dimension k (k << d)
    noise=0.1,           # gaussian noise std added in ambient space
    center=True,         # subtract feature means
    orthonormal=True,    # use orthonormal basis for projection
    shuffle=True,
    random_state=None,
    dtype=None,
    return_components=False,
):
    """
    Low-rank Gaussian subspace dataset:
      Z ~ N(0, I_k)
      X = Z @ W^T + eps,   where W is (d, k), eps ~ N(0, noise^2 I_d)

    Returns:
        X: (n_samples, n_features)
        (optional) W: (n_features, rank) projection basis (components)
        (optional) Z: (n_samples, rank) latent coordinates
    """
    if dtype is None:
        dtype = DTYPE
    if rank < 1 or rank > n_features:
        raise ValueError("rank must be in [1, n_features]")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    rng = _get_rng(random_state)

    # latent coords
    Z = rng.normal(0.0, 1.0, size=(n_samples, rank)).astype(dtype)

    # random projection matrix W (d x k)
    W = rng.normal(0.0, 1.0, size=(n_features, rank)).astype(dtype)

    if orthonormal:
        # Orthonormalize columns of W via QR
        # xp.linalg.qr returns Q shape (d, k) for reduced mode (depends on backend)
        Q, _ = xp.linalg.qr(W)
        W = Q[:, :rank].astype(dtype)

    X = Z @ W.T  # (n_samples, d)

    if noise and noise > 0:
        X = X + rng.normal(0.0, noise, size=X.shape).astype(dtype)

    if center:
        X = X - xp.mean(X, axis=0, keepdims=True)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        Z = Z[perm]

    if return_components:
        return X.astype(dtype), W.astype(dtype), Z.astype(dtype)
    return X.astype(dtype)