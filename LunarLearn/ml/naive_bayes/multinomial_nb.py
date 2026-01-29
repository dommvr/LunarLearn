import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.ml.naive_bayes.utils import _encode_labels
from LunarLearn.core import Tensor
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class MultinomialNB(Estimator, ClassifierMixin):
    def __init__(self, alpha: float = 1.0, eps: float = 1e-12):
        self.alpha = alpha
        self.eps = eps
        self.classes_ = None              # xp.ndarray, (C,)
        self.class_log_prior_ = None      # xp.ndarray, (C,)
        self.feature_log_prob_ = None     # xp.ndarray, (C, F)

    def _joint_log_likelihood(self, X: Tensor):
        if self.classes_ is None:
            raise RuntimeError("MultinomialNB not fitted.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_arr = X.data.astype(DTYPE, copy=False)     # (N, F)
        # joint log likelihood: log P(y) + sum_j x_j log P(feature_j|y)
        # = X @ log_prob^T + class_log_prior
        jll = X_arr @ self.feature_log_prob_.T       # (N, C)
        jll = jll + self.class_log_prior_[None, :]   # (N, C)
        return jll

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
            self.classes_ = classes
            y_enc = y_enc.astype("int64")

            n_samples, n_features = X_arr.shape
            n_classes = classes.shape[0]

            # One-hot: (N, C)
            Y = xp.zeros((n_samples, n_classes), dtype=DTYPE)
            Y[xp.arange(n_samples), y_enc] = 1.0

            # Class counts: (C,)
            class_count = Y.sum(axis=0)  # float

            # Feature counts per class: (C, F)
            feature_count = Y.T @ X_arr

            # Laplace smoothing
            smoothed_fc = feature_count + self.alpha
            smoothed_fc_sum = smoothed_fc.sum(axis=1, keepdims=True)  # (C, 1)

            self.feature_log_prob_ = xp.log(
                smoothed_fc / xp.maximum(smoothed_fc_sum, self.eps)
            )

            # class prior
            total_count = xp.maximum(class_count.sum(), 1.0)
            class_prior = class_count / total_count
            self.class_log_prior_ = xp.log(class_prior + self.eps)

        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            jll = self._joint_log_likelihood(X)      # (N, C)

            max_log = jll.max(axis=1, keepdims=True)
            exp_shifted = xp.exp(jll - max_log)
            probs = exp_shifted / xp.maximum(exp_shifted.sum(axis=1, keepdims=True), self.eps)

            return Tensor(probs.astype(DTYPE), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            jll = self._joint_log_likelihood(X)      # (N, C)
            enc_idx = jll.argmax(axis=1).astype("int64")
            labels = self.classes_[enc_idx]
            return Tensor(labels, dtype=DTYPE)