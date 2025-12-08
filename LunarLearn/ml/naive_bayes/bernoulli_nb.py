import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.ml.naive_bayes.utils import _encode_labels
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class BernoulliNB(Estimator, ClassifierMixin):
    """
    Bernoulli Naive Bayes classifier.

    Suitable for binary features (0/1). Optionally binarizes input > threshold.
    """

    def __init__(self, alpha: float = 1.0, binarize: float | None = 0.0, eps: float = 1e-9):
        self.alpha = alpha
        self.binarize = binarize
        self.eps = eps

        self.classes_ = None                 # xp.ndarray, (C,)
        self.class_log_prior_ = None         # xp.ndarray, (C,)
        self.feature_log_prob_ = None        # xp.ndarray, (C, F)
        self.feature_log_neg_prob_ = None    # xp.ndarray, (C, F)

    def _binarize_X(self, X_arr):
        if self.binarize is None:
            return X_arr
        return (X_arr > self.binarize).astype(DTYPE)
    
    def _joint_log_likelihood(self, X: Tensor):
        if self.classes_ is None:
            raise RuntimeError("BernoulliNB not fitted.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_arr = X.data.astype(DTYPE, copy=False)
        X_arr = self._binarize_X(X_arr)

        # X: (N, F)
        # p: (C, F)
        # jll_n,c = sum_j [ x_nj * log(p_cj) + (1 - x_nj) * log(1-p_cj) ] + log P(y=c)
        X_exp = X_arr[:, None, :]                          # (N, 1, F)
        flp = self.feature_log_prob_[None, :, :]           # (1, C, F)
        flnp = self.feature_log_neg_prob_[None, :, :]      # (1, C, F)

        jll = (X_exp * flp + (1.0 - X_exp) * flnp).sum(axis=2)  # (N, C)
        jll = jll + self.class_log_prior_[None, :]              # (N, C)
        return jll

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            X_arr = self._binarize_X(X_arr)

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

            # Feature counts (number of 1s per class): (C, F)
            feature_count = Y.T @ X_arr  # since X_arr is 0/1

            # n_c for each class: (C, 1)
            n_c = class_count[:, None]

            # P(x_j = 1 | y = c) with Laplace smoothing
            # p = (count_1 + alpha) / (n_c + 2 * alpha)
            p = (feature_count + self.alpha) / xp.maximum(n_c + 2 * self.alpha, self.eps)

            p = xp.clip(p, self.eps, 1.0 - self.eps)

            self.feature_log_prob_ = xp.log(p)
            self.feature_log_neg_prob_ = xp.log(1.0 - p)

            # class prior
            total_count = xp.maximum(class_count.sum(), 1.0)
            class_prior = class_count / total_count
            self.class_log_prior_ = xp.log(class_prior + self.eps)

        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            jll = self._joint_log_likelihood(X)

            max_log = jll.max(axis=1, keepdims=True)
            exp_shifted = xp.exp(jll - max_log)
            probs = exp_shifted / xp.maximum(exp_shifted.sum(axis=1, keepdims=True), self.eps)

            return Tensor(probs.astype(DTYPE), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            jll = self._joint_log_likelihood(X)
            enc_idx = jll.argmax(axis=1).astype("int64")
            labels = self.classes_[enc_idx]
            return Tensor(labels, dtype=DTYPE)