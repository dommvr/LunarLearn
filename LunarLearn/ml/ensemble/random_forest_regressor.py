import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.ml.tree import DecisionTreeRegressor
from LunarLearn.ml.ensemble.utils import _resolve_max_features
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class RandomForestRegressor(Estimator, RegressorMixin):
    """
    Random Forest regressor using DecisionTreeRegressor as base estimator.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int | None
        Maximum depth of each tree.
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node.
    max_features : int | float | str | None
        Number of features to consider when looking for the best split.
        Same semantics as in RandomForestClassifier.
    bootstrap : bool
        Whether to use bootstrap samples.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features="sqrt",
        bootstrap: bool = True,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap

        self.estimators_: list[DecisionTreeRegressor] = []
        self.features_: list[xp.ndarray] = []

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
            m_features = _resolve_max_features(n_features, self.max_features)

            self.estimators_ = []
            self.features_ = []

            for _ in range(self.n_estimators):
                if self.bootstrap:
                    indices = xp.random.randint(0, n_samples, size=(n_samples,))
                else:
                    indices = xp.arange(n_samples, dtype="int64")

                feat_idx = xp.random.choice(
                    n_features, size=(m_features,), replace=False
                )

                X_boot = X_arr[indices][:, feat_idx]
                y_boot = y_arr[indices]

                X_tensor = Tensor(X_boot, dtype=DTYPE)
                y_tensor = Tensor(y_boot, dtype=DTYPE)

                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                )
                tree.fit(X_tensor, y_tensor)

                self.estimators_.append(tree)
                self.features_.append(feat_idx)

            if not self.estimators_:
                raise RuntimeError("RandomForestRegressor fitting produced no estimators.")

        return self

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if not self.estimators_:
                raise RuntimeError("RandomForestRegressor not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]

            preds_sum = xp.zeros((n_samples,), dtype=DTYPE)

            for tree, feat_idx in zip(self.estimators_, self.features_):
                X_sub = X_arr[:, feat_idx]
                X_tensor = Tensor(X_sub, dtype=DTYPE)
                tree_preds = tree.predict(X_tensor)          # Tensor (N,)
                preds_sum += tree_preds.data.astype(DTYPE, copy=False).reshape(-1)

            preds_avg = preds_sum / float(len(self.estimators_))
            return Tensor(preds_avg, dtype=DTYPE)