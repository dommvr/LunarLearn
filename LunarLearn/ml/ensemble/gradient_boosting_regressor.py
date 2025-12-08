import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.ml.tree import DecisionTreeRegressor
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class GradientBoostingRegressor(Estimator, RegressorMixin):
    """
    Gradient Boosting Regressor (squared loss), using DecisionTreeRegressor as base learner.

    Model:
        F_0(x) = mean(y)
        For m = 1..M:
            r_m = y - F_{m-1}(x)         (negative gradient for MSE)
            fit tree h_m on (X, r_m)
            F_m(x) = F_{m-1}(x) + lr * h_m(x)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int | None = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.init_: float | None = None          # scalar bias
        self.estimators_: list[DecisionTreeRegressor] = []

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
                raise ValueError("Cannot fit GradientBoostingRegressor on empty data.")

            # Initial prediction: mean of y
            init = float(y_arr.mean())
            self.init_ = init

            # Current predictions F(x)
            F_current = xp.full_like(y_arr, init, dtype=DTYPE)

            self.estimators_ = []

            for _ in range(self.n_estimators):
                # negative gradient for MSE: r = y - F
                residual = y_arr - F_current

                X_tensor = Tensor(X_arr, dtype=DTYPE)
                r_tensor = Tensor(residual, dtype=DTYPE)

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                )
                tree.fit(X_tensor, r_tensor)
                self.estimators_.append(tree)

                # update F(x) with new tree
                pred = tree.predict(X_tensor).data.astype(DTYPE, copy=False).reshape(-1)
                F_current = F_current + self.learning_rate * pred

        return self

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.init_ is None:
                raise RuntimeError("GradientBoostingRegressor not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]

            # start from initial bias
            preds = xp.full((n_samples,), self.init_, dtype=DTYPE)

            if self.estimators_:
                X_tensor = Tensor(X_arr, dtype=DTYPE)
                for tree in self.estimators_:
                    step_pred = tree.predict(X_tensor).data.astype(DTYPE, copy=False).reshape(-1)
                    preds = preds + self.learning_rate * step_pred

            return Tensor(preds, dtype=DTYPE)