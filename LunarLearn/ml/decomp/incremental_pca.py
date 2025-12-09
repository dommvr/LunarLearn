import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, TransformMixin
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class IncrementalPCA(Estimator, TransformMixin):
    """
    Incremental PCA using a running covariance estimate.

    Maintains:
    - running mean
    - running (unnormalized) covariance matrix
    and recomputes eigen-decomposition after each batch.

    Parameters
    ----------
    n_components : int | None
        Number of components to keep.
        If None, keep all up to min(n_features, n_samples_seen).
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

        self.mean_: xp.ndarray | None = None               # (d,)
        self.cov_: xp.ndarray | None = None                # (d, d), unnormalized sum of outer products
        self.n_samples_seen_: int = 0

        self.components_: xp.ndarray | None = None          # (k, d)
        self.explained_variance_: xp.ndarray | None = None  # (k,)
        self.explained_variance_ratio_: xp.ndarray | None = None  # (k,)
        self.singular_values_: xp.ndarray | None = None     # (k,)
        self.n_components_: int | None = None
        self.n_features_: int | None = None

    def partial_fit(self, X: Tensor):
        """
        Update the PCA estimate with a new batch of samples.

        Parameters
        ----------
        X : Tensor of shape (n_samples_batch, n_features)
        """
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_batch, n_features = X_arr.shape

            if n_batch == 0:
                return self

            if self.n_samples_seen_ == 0:
                # first batch
                self.n_features_ = n_features
                mean_batch = X_arr.mean(axis=0)
                Xc = X_arr - mean_batch[None, :]
                cov = Xc.T @ Xc          # unnormalized covariance sum

                self.mean_ = mean_batch.astype(DTYPE, copy=False)
                self.cov_ = cov.astype(DTYPE, copy=False)
                self.n_samples_seen_ = n_batch
            else:
                if n_features != self.n_features_:
                    raise ValueError("Number of features has changed between calls to partial_fit.")

                n_old = self.n_samples_seen_
                n_new = n_old + n_batch

                mean_old = self.mean_
                cov_old = self.cov_

                mean_batch = X_arr.mean(axis=0)
                Xc_batch = X_arr - mean_batch[None, :]
                cov_batch = Xc_batch.T @ Xc_batch

                # new mean
                mean_new = (n_old * mean_old + n_batch * mean_batch) / float(n_new)

                # mean shift corrections for covariance
                # cov_new = cov_old + cov_batch
                #          + n_old * (mean_old - mean_new)(.)^T
                #          + n_batch * (mean_batch - mean_new)(.)^T
                delta_old = (mean_old - mean_new)[None, :]    # (1, d)
                delta_batch = (mean_batch - mean_new)[None, :]

                cov_new = cov_old + cov_batch
                cov_new += n_old * (delta_old.T @ delta_old)
                cov_new += n_batch * (delta_batch.T @ delta_batch)

                self.mean_ = mean_new.astype(DTYPE, copy=False)
                self.cov_ = cov_new.astype(DTYPE, copy=False)
                self.n_samples_seen_ = n_new

            # Recompute eigen-decomposition after each update
            n_samples_eff = self.n_samples_seen_
            if n_samples_eff <= 1:
                return self

            # normalized covariance
            cov_norm = self.cov_ / max(n_samples_eff - 1, 1)

            # symmetric, use eigh
            eigvals, eigvecs = xp.linalg.eigh(cov_norm)  # ascending
            # sort descending
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]  # columns

            if self.n_components is None:
                k = min(n_features, n_samples_eff)
            else:
                k = int(self.n_components)
                if k <= 0:
                    raise ValueError("n_components must be positive.")
                k = min(k, n_features, n_samples_eff)

            self.n_components_ = k
            self.components_ = eigvecs[:, :k].T.astype(DTYPE, copy=False)       # (k, d)
            self.explained_variance_ = eigvals[:k].astype(DTYPE, copy=False)
            total_var = eigvals.sum()
            if total_var > 0:
                self.explained_variance_ratio_ = (eigvals[:k] / total_var).astype(DTYPE, copy=False)
            else:
                self.explained_variance_ratio_ = xp.zeros_like(eigvals[:k], dtype=DTYPE)

            # singular values from eigenvalues: S = sqrt(ev * (n_samples_eff - 1))
            self.singular_values_ = xp.sqrt(
                xp.maximum(self.explained_variance_ * max(n_samples_eff - 1, 1), 0.0)
            ).astype(DTYPE, copy=False)

        return self

    def fit(self, X: Tensor):
        # Just a convenience wrapper: one-shot fit using partial_fit
        self.n_samples_seen_ = 0
        self.mean_ = None
        self.cov_ = None
        return self.partial_fit(X)

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.components_ is None or self.mean_ is None:
                raise RuntimeError("IncrementalPCA not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            X_centered = X_arr - self.mean_[None, :]
            Z = X_centered @ self.components_.T          # (n, k)
            return Tensor(Z.astype(DTYPE, copy=False), dtype=DTYPE)

    def inverse_transform(self, Z: Tensor) -> Tensor:
        with backend.no_grad():
            if self.components_ is None or self.mean_ is None:
                raise RuntimeError("IncrementalPCA not fitted.")

            Z_arr = Z.data.astype(DTYPE, copy=False)
            X_rec = Z_arr @ self.components_ + self.mean_[None, :]
            return Tensor(X_rec.astype(DTYPE, copy=False), dtype=DTYPE)