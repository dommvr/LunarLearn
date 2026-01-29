import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.ml.tree import DecisionTreeClassifier
from LunarLearn.ml.ensemble.utils import _encode_labels
from LunarLearn.core import Tensor, ops
from LunarLearn.core.tensor import ensure_tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


class AdaBoostClassifier(Estimator, ClassifierMixin):
    """
    AdaBoost classifier (SAMME-style, discrete boosting).

    Uses DecisionTreeClassifier as base estimator, trained on
    weighted-resampled data (since trees don't support sample weights).

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage applied to each estimator weight.
    max_depth : int
        Max depth of each base tree (depth=1 -> decision stumps).
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        eps: float = 1e-12,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.eps = eps

        self.classes_ = None                 # xp.ndarray, (C,)
        self.n_classes_ = None
        self.estimators_: list[DecisionTreeClassifier] = []
        self.estimator_weights_: xp.ndarray | None = None
        self.estimator_errors_: xp.ndarray | None = None

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            X = ensure_tensor(X)
            y = ensure_tensor(y)
            if self.n_estimators <= 0:
                raise ValueError("n_estimators must be > 0.")

            # Normalize shapes
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            y_arr = y.data

            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit AdaBoostClassifier on empty data.")

            classes, y_enc = _encode_labels(y)
            self.classes_ = classes
            self.n_classes_ = classes.shape[0]

            C = self.n_classes_
            if C < 2:
                raise ValueError("AdaBoostClassifier requires at least two classes.")

            # initial uniform sample weights
            sample_weight = xp.full((n_samples,), 1.0 / n_samples, dtype=DTYPE)

            self.estimators_ = []
            est_weights = []
            est_errors = []

            for m in range(self.n_estimators):
                # normalize weights
                sw_sum = float(sample_weight.sum())
                if sw_sum <= 0:
                    break
                sample_weight = sample_weight / sw_sum

                # weighted resampling of data
                indices = xp.random.choice(
                    n_samples,
                    size=n_samples,
                    replace=True,
                    p=sample_weight,
                )

                X_boot = X_arr[indices]
                y_boot = y_arr[indices]

                X_tensor = Tensor(X_boot, dtype=DTYPE)
                y_tensor = Tensor(y_boot, dtype=DTYPE)

                stump = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                )
                stump.fit(X_tensor, y_tensor)
                self.estimators_.append(stump)

                # predictions on original data
                X_full_tensor = Tensor(X_arr, dtype=DTYPE)
                y_pred = stump.predict(X_full_tensor).data

                # misclassification mask
                incorrect = (y_pred != y_arr).astype(DTYPE)

                # weighted error
                err_m = float((sample_weight * incorrect).sum())
                err_m = max(self.eps, min(err_m, 1.0 - self.eps))  # clamp

                # estimator weight (SAMME)
                # alpha_m = lr * ( log((1 - err)/err) + log(C - 1) )
                alpha_m = self.learning_rate * (
                    math.log((1.0 - err_m) / err_m) + math.log(C - 1.0)
                )

                est_weights.append(alpha_m)
                est_errors.append(err_m)

                # update sample weights: D_{m+1}(i) ∝ D_m(i) * exp(alpha_m * I[incorrect])
                sample_weight = sample_weight * xp.exp(alpha_m * incorrect)

                # if error is too high or too low, bail early
                if err_m >= 0.5:
                    # weak learner condition violated; stop adding more
                    break

            if not self.estimators_:
                raise RuntimeError("AdaBoostClassifier fitting produced no estimators.")

            self.estimator_weights_ = xp.array(est_weights, dtype=DTYPE)
            self.estimator_errors_ = xp.array(est_errors, dtype=DTYPE)

        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.classes_ is None or not self.estimators_:
                raise RuntimeError("AdaBoostClassifier not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]
            n_classes = self.n_classes_
            M = len(self.estimators_)

            scores = xp.zeros((n_samples, n_classes), dtype=DTYPE)

            for m, stump in enumerate(self.estimators_):
                alpha_m = float(self.estimator_weights_[m])
                X_tensor = Tensor(X_arr, dtype=DTYPE)
                y_pred = stump.predict(X_tensor).data  # original labels

                # encode predicted labels into 0..C-1
                enc_pred = xp.searchsorted(self.classes_, y_pred).astype("int64")

                # add alpha_m to the predicted class scores
                scores[xp.arange(n_samples), enc_pred] += alpha_m

            # turn scores into probabilities via softmax
            max_score = scores.max(axis=1, keepdims=True)
            exp_shifted = xp.exp(scores - max_score)
            probs = exp_shifted / xp.maximum(exp_shifted.sum(axis=1, keepdims=True), self.eps)

            return Tensor(probs.astype(DTYPE), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            probs = self.predict_proba(X)
            enc_idx = ops.argmax(probs, axis=1)            # Tensor of encoded indices
            enc_idx_arr = enc_idx.data.astype("int64")     # xp array
            labels = self.classes_[enc_idx_arr]
            return Tensor(labels, dtype=DTYPE)