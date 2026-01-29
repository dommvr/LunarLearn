import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.core import Tensor, ops
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class LinearRegression(Estimator, RegressorMixin):
    """
    Ordinary Least Squares Linear Regression.

    Parameters
    ----------
    fit_intercept : bool
        Whether to include an intercept term.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: Tensor, y: Tensor, eps: float = 1e-12):
        with backend.no_grad():
            X = ensure_tensor(X)
            y = ensure_tensor(y)
            # Ensure 2D X and 1D y
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)
            n_samples = X.shape[0]
            if self.fit_intercept:
                ones = ops.ones((n_samples, 1), dtype=DTYPE)
                X_ext = ops.concatenate([ones, X], axis=1)
            else:
                X_ext = X

            # Closed form: w = (X^T X)^-1 X^T y
            Xt = X_ext.T
            XtX = ops.matmul(Xt, X_ext)
            Xty = ops.matmul(Xt, y)

            # Add tiny regularization to avoid singular matrix issues
            XtX = XtX + eps * ops.eye(XtX.shape[0], dtype=X.dtype)

            w = ops.solve(XtX, Xty)

            if self.fit_intercept:
                self.intercept_ = w[0]
                self.coef_ = w[1:]
            else:
                self.intercept_ = ops.zeros((), dtype=X.dtype)
                self.coef_ = w

            return self

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return ops.matmul(X, self.coef_) + self.intercept_