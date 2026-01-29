import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, TransformMixin
from LunarLearn.core import Tensor
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class PCA(Estimator, TransformMixin):
    """
    Principal Component Analysis via SVD.

    Parameters
    ----------
    n_components : int | None
        Number of components to keep.
        If None, keep all min(n_samples, n_features) components.
    whiten : bool
        If True, scale components by 1 / sqrt(explained_variance_).
    """

    def __init__(self, n_components: int | None = None, whiten: bool = False):
        self.n_components = n_components
        self.whiten = whiten

        self.mean_: xp.ndarray | None = None               # (d,)
        self.components_: xp.ndarray | None = None          # (k, d)
        self.explained_variance_: xp.ndarray | None = None  # (k,)
        self.explained_variance_ratio_: xp.ndarray | None = None  # (k,)
        self.singular_values_: xp.ndarray | None = None     # (k,)
        self.n_components_: int | None = None
        self.n_features_: int | None = None
        self.n_samples_: int | None = None

    def fit(self, X: Tensor):
        with backend.no_grad():
            X = ensure_tensor(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit PCA on empty data.")

            self.n_features_ = n_features
            self.n_samples_ = n_samples

            # center data
            mean = X_arr.mean(axis=0)
            X_centered = X_arr - mean[None, :]

            # SVD: X_centered = U S V^T
            U, S, Vt = xp.linalg.svd(X_centered, full_matrices=False)

            # variance of each PC: S^2 / (n_samples - 1)
            var = (S ** 2) / max(n_samples - 1, 1)

            total_var = var.sum()

            if self.n_components is None:
                k = min(n_samples, n_features)
            else:
                k = int(self.n_components)
                if k <= 0:
                    raise ValueError("n_components must be positive.")
                k = min(k, min(n_samples, n_features))

            self.n_components_ = k
            self.mean_ = mean.astype(DTYPE, copy=False)
            self.components_ = Vt[:k].astype(DTYPE, copy=False)          # (k, d)
            self.singular_values_ = S[:k].astype(DTYPE, copy=False)
            self.explained_variance_ = var[:k].astype(DTYPE, copy=False)
            if total_var > 0:
                self.explained_variance_ratio_ = (var[:k] / total_var).astype(DTYPE, copy=False)
            else:
                self.explained_variance_ratio_ = xp.zeros_like(var[:k], dtype=DTYPE)

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.components_ is None or self.mean_ is None:
                raise RuntimeError("PCA not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            X_centered = X_arr - self.mean_[None, :]     # (n, d)

            # Project: X_centered @ components^T
            Z = X_centered @ self.components_.T          # (n, k)

            if self.whiten:
                # scale by 1 / sqrt(explained_variance_)
                ev = self.explained_variance_
                inv_std = 1.0 / xp.sqrt(xp.maximum(ev, 1e-12))
                Z = Z * inv_std[None, :]

            return Tensor(Z.astype(DTYPE, copy=False), dtype=DTYPE)

    def inverse_transform(self, Z: Tensor) -> Tensor:
        with backend.no_grad():
            Z = ensure_tensor(Z)
            if self.components_ is None or self.mean_ is None:
                raise RuntimeError("PCA not fitted.")

            Z_arr = Z.data.astype(DTYPE, copy=False)
            # Undo whitening if needed
            if self.whiten:
                ev = self.explained_variance_
                std = xp.sqrt(xp.maximum(ev, 1e-12))
                Z_arr = Z_arr * std[None, :]

            # Reconstruct: Z @ components + mean
            X_rec = Z_arr @ self.components_ + self.mean_[None, :]
            return Tensor(X_rec.astype(DTYPE, copy=False), dtype=DTYPE)