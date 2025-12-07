import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.core import Tensor, ops
from LunarLearn.amp import amp

xp = backend.xp
DTYPE = backend.DTYPE

class ElasticNetRegression(Estimator, RegressorMixin):
    def __init__(self,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: Tensor, y: Tensor, eps: float = 1e-12, use_amp: bool = True):
        with backend.no_grad():
            # Ensure 2D X and 1D y
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)
            with amp.autocast(enabled=use_amp):
                if self.fit_intercept:
                    self.intercept_ = y.mean()
                    y_centered = y - self.intercept_
                else:
                    self.intercept_ = ops.zeros((), dtype=X.dtype)
                    y_centered = y

                n_samples, n_features = X.shape
                w = ops.zeros(n_features, dtype=X.dtype)

                l1 = self.alpha * self.l1_ratio
                l2 = self.alpha * (1 - self.l1_ratio)

                X_col_norm_sq = (X ** 2).sum(axis=0) + l2

                for _ in range(self.max_iter):
                    w_old = w.clone()

                    for j in range(n_features):
                        y_pred = ops.matmul(X, w)
                        r_j = y_centered - (y_pred - X[:, j] * w[j])

                        rho_j = (X[:, j] * r_j).sum()

                        if rho_j < -l1 / 2:
                            w_j = (rho_j + l1 / 2) / (X_col_norm_sq[j] + eps)
                        elif rho_j > l1 / 2:
                            w_j = (rho_j - l1 / 2) / (X_col_norm_sq[j] + eps)
                        else:
                            w_j = ops.zeros((), dtype=X.dtype)

                        w[j] = w_j

                    if ops.norm(w - w_old, ord=1) < self.tol:
                        break

                self.coef_ = w
                return self
            
    def predict(self, X: Tensor, use_amp: bool = True) -> Tensor:
        with backend.no_grad():
            with amp.autocast(enabled=use_amp):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return ops.matmul(X, self.coef_) + self.intercept_