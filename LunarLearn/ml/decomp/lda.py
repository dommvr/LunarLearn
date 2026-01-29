import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin, TransformMixin
from LunarLearn.ml.decomp.utils import _encode_labels
from LunarLearn.ml.preprocessing import LabelEncoder
from LunarLearn.core import Tensor
from LunarLearn.core.tensor import ensure_tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


class LDA(Estimator, ClassifierMixin, TransformMixin):
    """
    Linear Discriminant Analysis (Fisher LDA / Gaussian discriminant).

    - Supervised dimensionality reduction:
        transform(X) projects onto up to (n_classes - 1) directions.
    - Classifier:
        assumes Gaussian class-conditional distributions
        with shared covariance, builds linear decision boundaries.

    Parameters
    ----------
    n_components : int | None
        Number of components to keep (<= n_classes - 1).
        If None, use n_classes - 1.
    reg_cov : float
        Regularization added to covariance diagonal.
    """

    def __init__(self, n_components: int | None = None, reg_cov: float = 1e-6):
        self.n_components = n_components
        self.reg_cov = reg_cov

        self.classes_: xp.ndarray | None = None          # (C,)
        self.priors_: xp.ndarray | None = None           # (C,)
        self.means_: xp.ndarray | None = None            # (C, d)
        self.cov_: xp.ndarray | None = None              # (d, d)
        self.coef_: xp.ndarray | None = None             # (C, d)
        self.intercept_: xp.ndarray | None = None        # (C,)

        self.scalings_: xp.ndarray | None = None         # (d, m)
        self.explained_variance_ratio_: xp.ndarray | None = None  # (m,)
        self.n_components_: int | None = None
        self.mean_: xp.ndarray | None = None             # overall mean, (d,)

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            X = ensure_tensor(X)
            y = ensure_tensor(y)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            classes, y_enc = _encode_labels(y)
            y_enc = y_enc.astype("int64")

            n_samples, n_features = X_arr.shape
            n_classes = classes.shape[0]

            if n_classes < 2:
                raise ValueError("LDA requires at least two classes.")
            if n_samples == 0:
                raise ValueError("Cannot fit LDA on empty data.")

            self.classes_ = classes

            # class counts and priors
            class_counts = xp.bincount(y_enc, minlength=n_classes).astype(DTYPE)
            priors = class_counts / float(class_counts.sum())
            self.priors_ = priors

            # class means
            means = xp.zeros((n_classes, n_features), dtype=DTYPE)
            for c in range(n_classes):
                mask = (y_enc == c)
                if not xp.any(mask):
                    raise ValueError(f"No samples for class index {c}.")
                Xc = X_arr[mask]
                means[c] = Xc.mean(axis=0)

            self.means_ = means

            # overall mean (weighted by priors)
            mean_overall = (priors[:, None] * means).sum(axis=0)
            self.mean_ = mean_overall.astype(DTYPE, copy=False)

            # pooled within-class covariance (unnormalized)
            cov = xp.zeros((n_features, n_features), dtype=DTYPE)
            for c in range(n_classes):
                mask = (y_enc == c)
                Xc = X_arr[mask]
                diff = Xc - means[c][None, :]
                cov += diff.T @ diff

            # normalize as covariance estimate
            denom = max(n_samples - n_classes, 1)
            cov /= float(denom)
            # regularize
            cov = cov + self.reg_cov * xp.eye(n_features, dtype=DTYPE)

            self.cov_ = cov.astype(DTYPE, copy=False)

            # classifier parameters (Gaussian with shared covariance)
            # w_c = Σ^{-1} μ_c,  b_c = -0.5 μ_c^T Σ^{-1} μ_c + log π_c
            cov_inv = xp.linalg.inv(cov)
            W = cov_inv @ means.T          # (d, C)
            W = W.T                        # (C, d)

            # compute intercepts
            # b_c = -0.5 μ_c^T Σ^{-1} μ_c + log π_c
            b = xp.empty((n_classes,), dtype=DTYPE)
            for c in range(n_classes):
                mc = means[c]        # (d,)
                wmc = cov_inv @ mc   # (d,)
                quad = mc @ wmc      # scalar
                b[c] = -0.5 * quad + math.log(float(priors[c]) + 1e-12)

            self.coef_ = W.astype(DTYPE, copy=False)
            self.intercept_ = b.astype(DTYPE, copy=False)

            # now compute LDA projection directions via generalized eigenproblem
            # between-class scatter: Sb = sum_c n_c (μ_c - μ)(μ_c - μ)^T
            Sb = xp.zeros((n_features, n_features), dtype=DTYPE)
            for c in range(n_classes):
                n_c = class_counts[c]
                mean_diff = (means[c] - mean_overall)[None, :]   # (1, d)
                Sb += n_c * (mean_diff.T @ mean_diff)

            # whitened between-class scatter: Σ^{-1/2} Sb Σ^{-1/2}
            eigvals_cov, eigvecs_cov = xp.linalg.eigh(self.cov_)  # cov is SPD-ish

            # avoid negative / tiny eigenvalues
            eigvals_cov = xp.maximum(eigvals_cov, 1e-12)
            inv_sqrt = 1.0 / xp.sqrt(eigvals_cov)
            # Σ^{-1/2} = Q diag(1/sqrt(λ)) Q^T
            cov_inv_sqrt = eigvecs_cov @ (inv_sqrt[:, None] * eigvecs_cov.T)

            Sb_tilde = cov_inv_sqrt @ Sb @ cov_inv_sqrt

            # symmetric, use eigh
            eigvals, eigvecs = xp.linalg.eigh(Sb_tilde)  # ascending
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

            # number of LDA components is at most C - 1
            max_lda_dim = n_classes - 1
            if self.n_components is None:
                m = max_lda_dim
            else:
                m = int(self.n_components)
                if m <= 0:
                    raise ValueError("n_components must be positive.")
                m = min(m, max_lda_dim, n_features)

            self.n_components_ = m

            if m > 0:
                # projection in original space: scalings = Σ^{-1/2} * eigenvectors
                scalings = cov_inv_sqrt @ eigvecs[:, :m]   # (d, m)
                self.scalings_ = scalings.astype(DTYPE, copy=False)

                # variance explained by each discriminant direction ~ eigenvalues
                # normalized to sum to 1 over first max_lda_dim
                ev_pos = xp.maximum(eigvals[:max_lda_dim], 0.0)
                total_ev = ev_pos.sum()
                if total_ev > 0:
                    ratio = (ev_pos[:m] / total_ev).astype(DTYPE, copy=False)
                else:
                    ratio = xp.zeros((m,), dtype=DTYPE)
                self.explained_variance_ratio_ = ratio
            else:
                self.scalings_ = None
                self.explained_variance_ratio_ = None

        return self

    def _decision_function(self, X_arr: xp.ndarray) -> xp.ndarray:
        """
        Compute linear discriminant scores: X @ coef_.T + intercept_.
        """
        scores = X_arr @ self.coef_.T + self.intercept_[None, :]
        return scores

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
                raise RuntimeError("LDA not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_arr = X.data.astype(DTYPE, copy=False)

            scores = self._decision_function(X_arr)      # (n, C)
            # convert scores (log posterior up to constant) to probabilities via softmax
            max_score = scores.max(axis=1, keepdims=True)
            exp_shifted = xp.exp(scores - max_score)
            probs = exp_shifted / xp.maximum(exp_shifted.sum(axis=1, keepdims=True), 1e-12)

            return Tensor(probs.astype(DTYPE, copy=False), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
                raise RuntimeError("LDA not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_arr = X.data.astype(DTYPE, copy=False)

            scores = self._decision_function(X_arr)
            enc_idx = scores.argmax(axis=1).astype("int64")
            labels = self.classes_[enc_idx]
            return Tensor(labels, dtype=DTYPE)

    def transform(self, X: Tensor) -> Tensor:
        """
        Project data onto LDA components (discriminant directions).

        Returns
        -------
        Tensor of shape (n_samples, n_components_)
        """
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.scalings_ is None or self.mean_ is None:
                raise RuntimeError("LDA not fitted or no components available.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            X_centered = X_arr - self.mean_[None, :]
            Z = X_centered @ self.scalings_           # (n, m)
            return Tensor(Z.astype(DTYPE, copy=False), dtype=DTYPE)