import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.ml.tree import DecisionTreeRegressor
from LunarLearn.ml.ensemble.utils import _encode_labels
from LunarLearn.core import Tensor, ops
from LunarLearn.core.tensor import ensure_tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


class GradientBoostingClassifier(Estimator, ClassifierMixin):
    """
    Gradient Boosting Classifier (binary, logistic loss), using DecisionTreeRegressor.

    Assumes binary classification with labels mapped to {0, 1}.

    Model:
        y in {0,1}, F(x) is logit.
        p(x) = sigmoid(F(x))
        Loss = - [ y log p + (1 - y) log (1 - p) ]

        F_0(x) = log(p / (1-p)), where p = mean(y)
        For m = 1..M:
            p_i = sigmoid(F_{m-1}(x_i))
            r_i = y_i - p_i              (negative gradient)
            fit tree h_m on (X, r)
            F_m(x) = F_{m-1}(x) + lr * h_m(x)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        eps: float = 1e-12,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.eps = eps

        self.init_: float | None = None             # scalar bias (logit)
        self.estimators_: list[DecisionTreeRegressor] = []
        self.classes_ = None                        # xp.ndarray, shape (2,)
        self.n_classes_ = None

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            X = ensure_tensor(X)
            y = ensure_tensor(y)
            if self.n_estimators <= 0:
                raise ValueError("n_estimators must be > 0.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            classes, y_enc = _encode_labels(y)
            self.classes_ = classes
            n_classes = classes.shape[0]
            if n_classes != 2:
                raise ValueError(
                    f"GradientBoostingClassifier currently supports only binary classification, "
                    f"got {n_classes} classes."
                )
            self.n_classes_ = n_classes

            y_bin = y_enc.astype(DTYPE)  # 0 or 1
            n_samples = X_arr.shape[0]
            if n_samples == 0:
                raise ValueError("Cannot fit GradientBoostingClassifier on empty data.")

            # Initial bias: log odds of positive class
            p = float(y_bin.mean())
            p = max(min(p, 1.0 - self.eps), self.eps)
            init_logit = math.log(p / (1.0 - p))
            self.init_ = init_logit

            F_current = xp.full_like(y_bin, init_logit, dtype=DTYPE)

            self.estimators_ = []

            for _ in range(self.n_estimators):
                # p_i = sigmoid(F_current)
                p_hat = ops.sigmoid(F_current).data

                # negative gradient of logloss wrt F: r = y - p_hat
                residual = y_bin - p_hat

                X_tensor = Tensor(X_arr, dtype=DTYPE)
                r_tensor = Tensor(residual, dtype=DTYPE)

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                )
                tree.fit(X_tensor, r_tensor)
                self.estimators_.append(tree)

                step_pred = tree.predict(X_tensor).data.astype(DTYPE, copy=False).reshape(-1)
                F_current = F_current + self.learning_rate * step_pred

        return self

    def _decision_function(self, X: Tensor) -> xp.ndarray:
        """
        Compute F(x) = initial logit + sum_m lr * h_m(x), returned as xp array.
        """
        if self.init_ is None:
            raise RuntimeError("GradientBoostingClassifier not fitted.")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_arr = X.data.astype(DTYPE, copy=False)
        n_samples = X_arr.shape[0]

        F = xp.full((n_samples,), self.init_, dtype=DTYPE)

        if self.estimators_:
            X_tensor = Tensor(X_arr, dtype=DTYPE)
            for tree in self.estimators_:
                step_pred = tree.predict(X_tensor).data.astype(DTYPE, copy=False).reshape(-1)
                F = F + self.learning_rate * step_pred

        return F

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.classes_ is None:
                raise RuntimeError("GradientBoostingClassifier not fitted.")

            F = self._decision_function(X)               # (N,)
            p_pos = ops.sigmoid(F).data                  # P(class 1)
            p_pos = xp.clip(p_pos, self.eps, 1.0 - self.eps)
            p_neg = 1.0 - p_pos

            # Order: encoded class 0, encoded class 1 -> self.classes_[0], self.classes_[1]
            probs = xp.stack([p_neg, p_pos], axis=1)     # (N, 2)
            return Tensor(probs.astype(DTYPE), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            probs = self.predict_proba(X)                # Tensor (N, 2)
            enc_idx = ops.argmax(probs, axis=1)          # encoded 0/1
            enc_idx_arr = enc_idx.data.astype("int64")
            labels = self.classes_[enc_idx_arr]
            return Tensor(labels, dtype=DTYPE)