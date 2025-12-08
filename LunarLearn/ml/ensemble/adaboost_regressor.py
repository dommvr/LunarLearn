import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.ml.tree import DecisionTreeRegressor
from LunarLearn.core import Tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


class AdaBoostRegressor(Estimator, RegressorMixin):
    """
    AdaBoost regressor (AdaBoost.R2-style) using DecisionTreeRegressor.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    learning_rate : float
        Shrinkage factor applied to estimator weights.
    max_depth : int
        Max depth of each base tree.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 3,
        eps: float = 1e-12,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.eps = eps

        self.estimators_: list[DecisionTreeRegressor] = []
        self.estimator_weights_: xp.ndarray | None = None

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            if self.n_estimators <= 0:
                raise ValueError("n_estimators must be > 0.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            y_arr = y.data.astype(DTYPE, copy=False)

            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit AdaBoostRegressor on empty data.")

            # initial uniform weights
            sample_weight = xp.full((n_samples,), 1.0 / n_samples, dtype=DTYPE)

            self.estimators_ = []
            est_weights = []

            for m in range(self.n_estimators):
                sw_sum = float(sample_weight.sum())
                if sw_sum <= 0:
                    break
                sample_weight = sample_weight / sw_sum

                # weighted resampling
                indices = xp.random.choice(
                    n_samples,
                    size=n_samples,
                    replace=True,
                    p=sample_weight,
                )

                X_boot = X_arr[indices]
                y_boot = y_arr[indices]

                X_tensor_boot = Tensor(X_boot, dtype=DTYPE)
                y_tensor_boot = Tensor(y_boot, dtype=DTYPE)

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                )
                tree.fit(X_tensor_boot, y_tensor_boot)
                self.estimators_.append(tree)

                # predictions on full data
                X_full_tensor = Tensor(X_arr, dtype=DTYPE)
                y_pred = tree.predict(X_full_tensor).data.astype(DTYPE, copy=False).reshape(-1)

                # absolute errors
                err = xp.abs(y_arr - y_pred)       # (n,)
                max_err = float(err.max())

                if max_err <= 0:
                    # perfect predictions; stop early
                    alpha_m = 1.0
                    est_weights.append(alpha_m)
                    break

                # normalized errors in [0, 1]
                err_norm = err / max_err

                # weighted error
                err_m = float((sample_weight * err_norm).sum())
                err_m = max(self.eps, min(err_m, 1.0 - self.eps))

                # beta and estimator weight (AdaBoost.R2)
                beta_m = err_m / (1.0 - err_m)
                alpha_m = self.learning_rate * math.log(1.0 / beta_m)
                est_weights.append(alpha_m)

                # update sample weights: w_i ∝ w_i * beta_m^{1 - err_norm_i}
                sample_weight = sample_weight * xp.power(beta_m, (1.0 - err_norm))

                # if error too bad, stop
                if err_m >= 0.5:
                    break

            if not self.estimators_:
                raise RuntimeError("AdaBoostRegressor fitting produced no estimators.")

            self.estimator_weights_ = xp.array(est_weights, dtype=DTYPE)

        return self

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if not self.estimators_ or self.estimator_weights_ is None:
                raise RuntimeError("AdaBoostRegressor not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]
            M = len(self.estimators_)

            X_tensor = Tensor(X_arr, dtype=DTYPE)

            # collect predictions from all estimators
            preds_all = xp.zeros((M, n_samples), dtype=DTYPE)
            for m, tree in enumerate(self.estimators_):
                y_pred = tree.predict(X_tensor).data.astype(DTYPE, copy=False).reshape(-1)
                preds_all[m] = y_pred

            # weighted average over estimators: sum_m alpha_m * f_m(x) / sum_m alpha_m
            weights = self.estimator_weights_.reshape(-1, 1)   # (M, 1)
            weighted_sum = (weights * preds_all).sum(axis=0)   # (n,)
            total_weight = max(float(weights.sum()), self.eps)

            preds = weighted_sum / total_weight
            return Tensor(preds, dtype=DTYPE)