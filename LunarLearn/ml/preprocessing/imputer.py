import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class Imputer(Estimator):
    """
    Simple strategy for imputing missing values (NaNs).

    Parameters
    ----------
    strategy : {"mean", "median", "most_frequent", "constant"}
        Imputation strategy.
    fill_value : float | int | None
        When strategy="constant", this value is used to replace NaNs.
        Ignored otherwise.
    """

    def __init__(self, strategy: str = "mean", fill_value=None):
        if strategy not in ("mean", "median", "most_frequent", "constant"):
            raise ValueError(
                "strategy must be one of {'mean', 'median', 'most_frequent', 'constant'}."
            )
        self.strategy = strategy
        self.fill_value = fill_value

        self.statistics_: xp.ndarray | None = None  # (d,)
        self.n_features_: int | None = None

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit SimpleImputer on empty data.")

            self.n_features_ = n_features

            stats = xp.empty((n_features,), dtype=DTYPE)

            if self.strategy == "constant":
                # Fill each feature with the same constant
                if self.fill_value is None:
                    raise ValueError("fill_value must be set when strategy='constant'.")
                stats.fill(self.fill_value)
            else:
                for j in range(n_features):
                    col = X_arr[:, j]
                    mask = ~xp.isnan(col)
                    if not xp.any(mask):
                        # all values missing, fallback to 0
                        stats[j] = 0.0
                        continue

                    valid = col[mask]

                    if self.strategy == "mean":
                        stats[j] = valid.mean()
                    elif self.strategy == "median":
                        stats[j] = xp.median(valid)
                    elif self.strategy == "most_frequent":
                        # mode via unique + counts
                        uniq, counts = xp.unique(valid, return_counts=True)
                        idx = counts.argmax()
                        stats[j] = uniq[idx]

            self.statistics_ = stats

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.statistics_ is None:
                raise RuntimeError("SimpleImputer not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_features_ or 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            if X_arr.shape[1] != self.n_features_:
                raise ValueError(
                    f"Expected {self.n_features_} features, got {X_arr.shape[1]}."
                )

            X_out = X_arr.copy()
            # replace NaNs column-wise
            for j in range(self.n_features_):
                col = X_out[:, j]
                mask = xp.isnan(col)
                if xp.any(mask):
                    col[mask] = self.statistics_[j]
                    X_out[:, j] = col

            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)