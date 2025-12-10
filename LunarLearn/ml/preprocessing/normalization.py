import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, TransformMixin
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.xp


class Normalizer(Estimator, TransformMixin):
    """
    Normalize samples individually to unit norm.

    For each sample i:
        X[i, :] <- X[i, :] / ||X[i, :]||_norm

    Parameters
    ----------
    norm : {"l1", "l2", "max"}
        Which norm to use for normalization.
    """

    def __init__(self, norm: str = "l2"):
        if norm not in ("l1", "l2", "max"):
            raise ValueError("norm must be one of {'l1', 'l2', 'max'}.")
        self.norm = norm

    def fit(self, X: Tensor):
        # Stateless transformer, just return self
        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(1, -1)

            X_arr = X.data.astype(DTYPE, copy=False)
            if self.norm == "l2":
                norms = xp.sqrt(xp.maximum((X_arr ** 2).sum(axis=1, keepdims=True), 1e-12))
            elif self.norm == "l1":
                norms = xp.maximum(xp.abs(X_arr).sum(axis=1, keepdims=True), 1e-12)
            else:  # "max"
                norms = xp.maximum(xp.abs(X_arr).max(axis=1, keepdims=True), 1e-12)

            X_out = X_arr / norms
            return Tensor(X_out.astype(DTYPE, copy=False), dtype=DTYPE)
        

class Binarizer(Estimator, TransformMixin):
    """
    Binarize data based on a threshold.

    For each value x:
        x <- 1 if x > threshold else 0

    Parameters
    ----------
    threshold : float
        Values greater than threshold are mapped to 1, others to 0.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = float(threshold)

    def fit(self, X: Tensor):
        # Stateless
        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X_arr = X.data.astype(DTYPE, copy=False)
            X_out = (X_arr > self.threshold).astype(DTYPE)
            return Tensor(X_out, dtype=DTYPE)