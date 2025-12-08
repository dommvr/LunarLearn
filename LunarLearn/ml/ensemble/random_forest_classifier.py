import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.ml.tree import DecisionTreeClassifier
from LunarLearn.ml.ensemble.utils import _resolve_max_features
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE


class RandomForestClassifier(Estimator, ClassifierMixin):
    """
    Random Forest classifier using DecisionTreeClassifier as base estimator.

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
        - None: all features
        - int: that many features
        - float in (0, 1]: fraction of features
        - "sqrt": sqrt(n_features)
        - "log2": log2(n_features)
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

        self.estimators_: list[DecisionTreeClassifier] = []
        self.features_: list[xp.ndarray] = []  # feature indices per tree
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
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
            m_features = _resolve_max_features(n_features, self.max_features)

            self.estimators_ = []
            self.features_ = []

            for _ in range(self.n_estimators):
                # Bootstrap sample indices
                if self.bootstrap:
                    indices = xp.random.randint(0, n_samples, size=(n_samples,))
                else:
                    indices = xp.arange(n_samples, dtype="int64")

                # Feature subsampling without replacement
                feat_idx = xp.random.choice(
                    n_features, size=(m_features,), replace=False
                )

                X_boot = X_arr[indices][:, feat_idx]
                y_boot = y_arr[indices]

                # Wrap into Tensors
                X_tensor = Tensor(X_boot, dtype=DTYPE)
                y_tensor = Tensor(y_boot, dtype=DTYPE)

                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                )
                tree.fit(X_tensor, y_tensor)

                self.estimators_.append(tree)
                self.features_.append(feat_idx)

            # Expose classes_ based on first estimator
            if not self.estimators_:
                raise RuntimeError("RandomForestClassifier fitting produced no estimators.")

            self.classes_ = self.estimators_[0].classes_
            self.n_classes_ = self.classes_.shape[0]

        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if not self.estimators_ or self.classes_ is None:
                raise RuntimeError("RandomForestClassifier not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]
            n_classes = self.n_classes_

            # Accumulate probabilities
            probs_sum = xp.zeros((n_samples, n_classes), dtype=DTYPE)

            for tree, feat_idx in zip(self.estimators_, self.features_):
                X_sub = X_arr[:, feat_idx]
                X_tensor = Tensor(X_sub, dtype=DTYPE)
                tree_probs = tree.predict_proba(X_tensor)   # Tensor (N, C)
                probs_sum += tree_probs.data.astype(DTYPE, copy=False)

            probs_avg = probs_sum / float(len(self.estimators_))
            return Tensor(probs_avg, dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            probs = self.predict_proba(X)
            enc_idx = ops.argmax(probs, axis=1)                 # Tensor
            enc_idx_arr = enc_idx.data.astype("int64")          # xp array
            labels = self.classes_[enc_idx_arr]
            return Tensor(labels, dtype=DTYPE)