import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class LinearSVR(Estimator, RegressorMixin):
    """
    Linear Support Vector Regression (ε-SVR) with L2 regularization.

    Optimization:
        Minimize over w, b:
            L = 0.5 * reg * ||w||^2 + mean_i max(0, |f(x_i) - y_i| - eps)

    where:
        f(x_i) = w^T x_i + b

    Gradients are computed in batch and updated via gradient descent.

    Parameters
    ----------
    reg : float
        L2 regularization coefficient (lambda). Larger -> stronger regularization.
    eps : float
        Epsilon-insensitive zone around target values.
    lr : float
        Learning rate for gradient updates.
    n_epochs : int
        Number of passes over the full dataset.
    fit_intercept : bool
        Whether to learn an intercept term.
    """

    def __init__(
        self,
        reg: float = 1e-4,
        eps: float = 0.1,
        lr: float = 1e-2,
        n_epochs: int = 1000,
        fit_intercept: bool = True,
    ):
        self.reg = float(reg)
        self.eps = float(eps)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.fit_intercept = bool(fit_intercept)

        self.coef_: xp.ndarray | None = None           # (d,)
        self.intercept_: float | None = None
        self.n_features_: int | None = None

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            y_arr = y.data.astype(DTYPE, copy=False)

            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit LinearSVR on empty data.")

            self.n_features_ = n_features

            reg = self.reg
            eps = self.eps
            lr = self.lr

            # init params
            w = xp.zeros((n_features,), dtype=DTYPE)
            b = 0.0

            for _ in range(self.n_epochs):
                # predictions
                f = X_arr @ w
                if self.fit_intercept:
                    f = f + b

                # errors
                e = f - y_arr   # (n,)
                abs_e = xp.abs(e)

                # mask where |e| > eps
                mask = abs_e > eps
                if not xp.any(mask):
                    # only regularization gradient
                    grad_w = reg * w
                    grad_b = 0.0
                else:
                    X_m = X_arr[mask]      # (m, d)
                    e_m = e[mask]          # (m,)
                    m = X_m.shape[0]

                    # derivative of max(0, |e| - eps) wrt f:
                    # sign(e) for |e| > eps, 0 otherwise
                    grad_loss_f = xp.sign(e_m)    # (m,)

                    # gradient wrt w, b:
                    # dL/dw = reg * w + (X_m^T @ grad_loss_f) / n
                    # dL/db = grad_loss_f.sum() / n
                    grad_w_loss = (X_m.T @ grad_loss_f) / float(n_samples)
                    grad_b_loss = grad_loss_f.sum() / float(n_samples)

                    grad_w = reg * w + grad_w_loss
                    grad_b = grad_b_loss if self.fit_intercept else 0.0

                # gradient descent step
                w = w - lr * grad_w
                if self.fit_intercept:
                    b = b - lr * grad_b

            self.coef_ = w
            self.intercept_ = float(b)

        return self

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.coef_ is None:
                raise RuntimeError("LinearSVR not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_features_ or 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            f = X_arr @ self.coef_
            if self.fit_intercept:
                f = f + self.intercept_

            return Tensor(f.astype(DTYPE, copy=False), dtype=DTYPE)