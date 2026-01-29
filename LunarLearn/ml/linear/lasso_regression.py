import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.core import Tensor, ops
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class LassoRegression(Estimator, RegressorMixin):
    def __init__(self,
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
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
            if self.fit_intercept:
                self.intercept_ = y.mean()
                y_centered = y - self.intercept_
            else:
                self.intercept_ = ops.zeros((), dtype=X.dtype)
                y_centered = y

            n_samples, n_features = X.shape
            w = ops.zeros(n_features, dtype=X.dtype)

            X_col_norm_sq = (X ** 2).sum(axis=0)

            for _ in range(self.max_iter):
                w_old = w.clone()

                for j in range(n_features):
                    # Residual excluding feature j
                    y_pred = ops.matmul(X, w)
                    r_j = y_centered - (y_pred - X[:, j] * w[j])

                    rho_j = (X[:, j] * r_j).sum()

                    if rho_j < -self.alpha / 2:
                        w_j = (rho_j + self.alpha / 2) / (X_col_norm_sq[j] + eps)
                    elif rho_j > self.alpha / 2:
                        w_j = (rho_j - self.alpha / 2) / (X_col_norm_sq[j] + eps)
                    else:
                        w_j = ops.zeros((), dtype=X.dtype)

                    w[j] = w_j

                if ops.norm(w - w_old, ord=1) < self.tol:
                    break

            self.coef_ = w
            return self
            
    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return ops.matmul(X, self.coef_) + self.intercept_