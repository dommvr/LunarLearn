import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, TransformMixin
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class StandardScaler(Estimator, TransformMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    For each feature j:
        X[:, j] <- (X[:, j] - mean_j) / scale_j

    where scale_j is the standard deviation (or 1 if with_std=False).

    Parameters
    ----------
    with_mean : bool
        Whether to center data before scaling.
    with_std : bool
        Whether to scale data to unit variance.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)

        self.mean_: xp.ndarray | None = None   # (d,)
        self.scale_: xp.ndarray | None = None  # (d,)
        self.n_features_: int | None = None

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit StandardScaler on empty data.")

            self.n_features_ = n_features

            if self.with_mean:
                mean = X_arr.mean(axis=0)
            else:
                mean = xp.zeros((n_features,), dtype=DTYPE)

            if self.with_std:
                # variance with ddof=0 (population)
                var = X_arr.var(axis=0)
                scale = xp.sqrt(xp.maximum(var, 1e-12))
            else:
                scale = xp.ones((n_features,), dtype=DTYPE)

            self.mean_ = mean.astype(DTYPE, copy=False)
            self.scale_ = scale.astype(DTYPE, copy=False)

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError("StandardScaler not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_features_ or 1)

            X_arr = X.data.astype(DTYPE, copy=False)

            X_out = X_arr
            if self.with_mean:
                X_out = X_out - self.mean_[None, :]

            if self.with_std:
                X_out = X_out / self.scale_[None, :]

            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)

    def inverse_transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError("StandardScaler not fitted.")

            X_arr = X.data.astype(DTYPE, copy=False)
            X_out = X_arr

            if self.with_std:
                X_out = X_out * self.scale_[None, :]

            if self.with_mean:
                X_out = X_out + self.mean_[None, :]

            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)


class MinMaxScaler(Estimator, TransformMixin):
    """
    Transform features by scaling each feature to a given range.

    For each feature j:
        X[:, j] <- X_min + (X[:, j] - data_min_j) * (X_max - X_min) / (data_max_j - data_min_j)

    Parameters
    ----------
    feature_range : tuple[float, float]
        Desired output range (min, max).
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        if len(feature_range) != 2:
            raise ValueError("feature_range must be a tuple of (min, max).")
        self.feature_range = (float(feature_range[0]), float(feature_range[1]))

        self.data_min_: xp.ndarray | None = None   # (d,)
        self.data_max_: xp.ndarray | None = None   # (d,)
        self.data_range_: xp.ndarray | None = None # (d,)
        self.scale_: xp.ndarray | None = None      # (d,)
        self.min_: xp.ndarray | None = None        # (d,)
        self.n_features_: int | None = None

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit MinMaxScaler on empty data.")

            self.n_features_ = n_features

            data_min = X_arr.min(axis=0)
            data_max = X_arr.max(axis=0)
            data_range = data_max - data_min

            # avoid zero division
            data_range = xp.where(data_range == 0, 1.0, data_range)

            f_min, f_max = self.feature_range
            scale = (f_max - f_min) / data_range
            min_offset = f_min - data_min * scale

            self.data_min_ = data_min.astype(DTYPE, copy=False)
            self.data_max_ = data_max.astype(DTYPE, copy=False)
            self.data_range_ = data_range.astype(DTYPE, copy=False)
            self.scale_ = scale.astype(DTYPE, copy=False)
            self.min_ = min_offset.astype(DTYPE, copy=False)

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.scale_ is None or self.min_ is None:
                raise RuntimeError("MinMaxScaler not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_features_ or 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            X_out = X_arr * self.scale_[None, :] + self.min_[None, :]
            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)

    def inverse_transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.scale_ is None or self.min_ is None:
                raise RuntimeError("MinMaxScaler not fitted.")

            X_arr = X.data.astype(DTYPE, copy=False)
            # undo: X_scaled = X * scale + min  =>  X = (X_scaled - min) / scale
            X_out = (X_arr - self.min_[None, :]) / self.scale_[None, :]
            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)


class MaxAbsScaler(Estimator, TransformMixin):
    """
    Scale each feature by its maximum absolute value.

    For each feature j:
        X[:, j] <- X[:, j] / max(|X[:, j]|)

    This preserves sparsity patterns (no centering).

    Parameters
    ----------
    None
    """

    def __init__(self):
        self.max_abs_: xp.ndarray | None = None   # (d,)
        self.n_features_: int | None = None

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit MaxAbsScaler on empty data.")

            self.n_features_ = n_features

            max_abs = xp.max(xp.abs(X_arr), axis=0)
            max_abs = xp.where(max_abs == 0, 1.0, max_abs)

            self.max_abs_ = max_abs.astype(DTYPE, copy=False)

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.max_abs_ is None:
                raise RuntimeError("MaxAbsScaler not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_features_ or 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            X_out = X_arr / self.max_abs_[None, :]
            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)

    def inverse_transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.max_abs_ is None:
                raise RuntimeError("MaxAbsScaler not fitted.")

            X_arr = X.data.astype(DTYPE, copy=False)
            X_out = X_arr * self.max_abs_[None, :]
            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)


class RobustScaler(Estimator, TransformMixin):
    """
    Scale features using statistics robust to outliers.

    For each feature j:
        X[:, j] <- (X[:, j] - median_j) / IQR_j

    where IQR_j = q75_j - q25_j.

    Parameters
    ----------
    with_centering : bool
        Whether to subtract the median.
    with_scaling : bool
        Whether to scale by IQR.
    quantile_range : tuple[float, float]
        Quantile range used to compute IQR (default (25.0, 75.0)).
    """

    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: tuple[float, float] = (25.0, 75.0),
    ):
        self.with_centering = bool(with_centering)
        self.with_scaling = bool(with_scaling)

        if len(quantile_range) != 2:
            raise ValueError("quantile_range must be a tuple (q_min, q_max).")
        q_min, q_max = quantile_range
        if not (0.0 <= q_min < q_max <= 100.0):
            raise ValueError("quantile_range values must satisfy 0 <= q_min < q_max <= 100.")
        self.quantile_range = (float(q_min), float(q_max))

        self.center_: xp.ndarray | None = None   # (d,)
        self.scale_: xp.ndarray | None = None    # (d,)
        self.n_features_: int | None = None

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit RobustScaler on empty data.")

            self.n_features_ = n_features

            q_min, q_max = self.quantile_range

            # percentiles along axis=0
            q = xp.percentile(X_arr, [q_min, q_max], axis=0)
            q_low = q[0]
            q_high = q[1]

            median = xp.median(X_arr, axis=0)

            iqr = q_high - q_low
            # avoid zero IQR
            iqr = xp.where(iqr == 0, 1.0, iqr)

            if self.with_centering:
                center = median
            else:
                center = xp.zeros((n_features,), dtype=DTYPE)

            if self.with_scaling:
                scale = iqr
            else:
                scale = xp.ones((n_features,), dtype=DTYPE)

            self.center_ = center.astype(DTYPE, copy=False)
            self.scale_ = scale.astype(DTYPE, copy=False)

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.center_ is None or self.scale_ is None:
                raise RuntimeError("RobustScaler not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_features_ or 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            X_out = X_arr

            if self.with_centering:
                X_out = X_out - self.center_[None, :]
            if self.with_scaling:
                X_out = X_out / self.scale_[None, :]

            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)

    def inverse_transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.center_ is None or self.scale_ is None:
                raise RuntimeError("RobustScaler not fitted.")

            X_arr = X.data.astype(DTYPE, copy=False)
            X_out = X_arr

            if self.with_scaling:
                X_out = X_out * self.scale_[None, :]
            if self.with_centering:
                X_out = X_out + self.center_[None, :]

            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)