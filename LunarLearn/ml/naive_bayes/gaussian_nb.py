import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.ml.naive_bayes.utils import _encode_labels
from LunarLearn.core import Tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


class GaussianNB(Estimator, ClassifierMixin):
    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        self.classes_ = None          # xp.ndarray, shape (n_classes,)
        self.class_prior_ = None      # xp.ndarray, shape (n_classes,)
        self.theta_ = None            # xp.ndarray, shape (n_classes, n_features)
        self.var_ = None              # xp.ndarray, shape (n_classes, n_features)

    def _joint_log_likelihood(self, X: Tensor):
        """
        Compute joint log likelihood log P(x, y) for each class.

        Returns:
            xp.ndarray of shape (n_samples, n_classes)
        """
        if self.classes_ is None:
            raise RuntimeError("GaussianNB not fitted.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_arr = X.data.astype(DTYPE, copy=False)
        n_samples, n_features = X_arr.shape

        theta = self.theta_              # (C, F)
        var = self.var_                  # (C, F)
        class_log_prior = xp.log(self.class_prior_ + self.eps)  # (C,)

        # X: (N, F), theta/var: (C, F)
        # Broadcast to (N, C, F)
        X_exp = X_arr[:, None, :]                 # (N, 1, F)
        mean_exp = theta[None, :, :]              # (1, C, F)
        var_exp = var[None, :, :]                 # (1, C, F)

        # log P(x|y=c) for Gaussian
        # = -0.5 * sum_j [ log(2πσ^2) + (x_j - μ_j)^2 / σ^2 ]
        log_var = xp.log(2.0 * math.pi * var_exp)
        sq_term = (X_exp - mean_exp) ** 2 / var_exp
        log_prob = -0.5 * (log_var + sq_term).sum(axis=2)    # (N, C)

        # add log prior
        jll = log_prob + class_log_prior[None, :]            # (N, C)
        return jll

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            # Ensure shapes
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)

            # Encode labels
            classes, y_enc = _encode_labels(y)
            self.classes_ = classes
            y_enc = y_enc.astype("int64")

            n_samples, n_features = X_arr.shape
            n_classes = classes.shape[0]

            # One-hot: (N, C)
            Y = xp.zeros((n_samples, n_classes), dtype=DTYPE)
            Y[xp.arange(n_samples), y_enc] = 1.0

            # Class counts: (C,)
            class_count = Y.sum(axis=0)  # float, but fine

            # Sums per class: (C, F)
            sum_x = Y.T @ X_arr           # (C, F)
            sum_x2 = Y.T @ (X_arr ** 2)   # (C, F)

            # Avoid division by zero
            denom = xp.maximum(class_count[:, None], 1.0)

            theta = sum_x / denom
            ex2 = sum_x2 / denom
            var = ex2 - theta ** 2
            var = var + self.eps

            self.theta_ = theta.astype(DTYPE)
            self.var_ = var.astype(DTYPE)

            # Prior
            total_count = xp.maximum(class_count.sum(), 1.0)
            class_prior = class_count / total_count
            self.class_prior_ = class_prior.astype(DTYPE)

        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            jll = self._joint_log_likelihood(X)              # (N, C)

            # softmax over classes
            max_log = jll.max(axis=1, keepdims=True)
            exp_shifted = xp.exp(jll - max_log)
            probs = exp_shifted / xp.maximum(
                exp_shifted.sum(axis=1, keepdims=True), self.eps
            )

            return Tensor(probs.astype(DTYPE), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            jll = self._joint_log_likelihood(X)              # (N, C)
            enc_idx = jll.argmax(axis=1).astype("int64")     # xp array
            labels = self.classes_[enc_idx]                  # original labels
            return Tensor(labels, dtype=DTYPE)