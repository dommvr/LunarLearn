import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.ml.svm.utils import _encode_labels
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class LinearSVC(Estimator, ClassifierMixin):
    """
    Linear Support Vector Classifier (multi-class via one-vs-rest).

    Optimization:
        Minimize over w, b (per class):
            L = 0.5 * reg * ||w||^2 + mean_i max(0, 1 - y_i * (w^T x_i + b))

    where y_i ∈ {+1, -1} for each one-vs-rest problem.

    Training is done by batch gradient descent.

    Parameters
    ----------
    reg : float
        L2 regularization coefficient (lambda). Larger -> stronger regularization.
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
        lr: float = 1e-2,
        n_epochs: int = 1000,
        fit_intercept: bool = True,
    ):
        self.reg = float(reg)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.fit_intercept = bool(fit_intercept)

        self.classes_: xp.ndarray | None = None        # (C,)
        self.coef_: xp.ndarray | None = None           # (C, d)
        self.intercept_: xp.ndarray | None = None      # (C,)
        self.n_features_: int | None = None

    def _fit_binary_ovr(
        self,
        X_arr: xp.ndarray,
        y_bin: xp.ndarray,
    ) -> tuple[xp.ndarray, float]:
        """
        Fit a binary linear SVM for y_bin ∈ {+1, -1} using batch gradient descent.

        Returns
        -------
        w : xp.ndarray, shape (d,)
        b : float
        """
        n_samples, n_features = X_arr.shape
        reg = self.reg
        lr = self.lr

        # initialize
        w = xp.zeros((n_features,), dtype=DTYPE)
        b = 0.0

        # ensure y_bin is float +-1
        yb = y_bin.astype(DTYPE, copy=False)

        for _ in range(self.n_epochs):
            # scores: (n,)
            scores = X_arr @ w
            if self.fit_intercept:
                scores = scores + b

            # margins: y_i * f(x_i)
            margins = yb * scores  # (n,)

            # hinge loss active where margin < 1
            mask = margins < 1.0
            if not xp.any(mask):
                # only regularization gradient
                grad_w = reg * w
                grad_b = 0.0
            else:
                X_m = X_arr[mask]        # (m, d)
                y_m = yb[mask]           # (m,)
                m = X_m.shape[0]
                # gradient of mean hinge term: -1/n * sum_{margin<1} y_i x_i
                grad_hinge_w = -(X_m.T @ y_m) / float(n_samples)
                grad_hinge_b = -y_m.sum() / float(n_samples)

                grad_w = reg * w + grad_hinge_w
                grad_b = grad_hinge_b if self.fit_intercept else 0.0

            # gradient descent update
            w = w - lr * grad_w
            if self.fit_intercept:
                b = b - lr * grad_b

        return w, float(b)

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            # normalize shapes
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit LinearSVC on empty data.")

            self.n_features_ = n_features

            classes, y_enc = _encode_labels(y)
            self.classes_ = classes
            y_enc = y_enc.astype("int64")
            n_classes = classes.shape[0]

            if n_classes == 1:
                raise ValueError("LinearSVC requires at least two classes.")

            # For binary case, we can just train one classifier
            if n_classes == 2:
                # map to +1/-1
                y_bin = xp.where(y_enc == 1, 1.0, -1.0)
                w, b = self._fit_binary_ovr(X_arr, y_bin)
                self.coef_ = w[None, :]                        # (1, d)
                self.intercept_ = xp.array([b], dtype=DTYPE)   # (1,)
            else:
                # one-vs-rest: train C binary classifiers
                coef = xp.zeros((n_classes, n_features), dtype=DTYPE)
                intercept = xp.zeros((n_classes,), dtype=DTYPE)

                for c in range(n_classes):
                    # y_bin = +1 for class c, -1 for rest
                    y_bin = xp.where(y_enc == c, 1.0, -1.0)
                    w, b = self._fit_binary_ovr(X_arr, y_bin)
                    coef[c] = w
                    intercept[c] = b

                self.coef_ = coef
                self.intercept_ = intercept

        return self

    def decision_function(self, X: Tensor) -> Tensor:
        """
        Compute decision scores for each class.

        Returns
        -------
        Tensor of shape (n_samples,) for binary or (n_samples, n_classes) for multi-class.
        """
        with backend.no_grad():
            if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
                raise RuntimeError("LinearSVC not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_features_ or 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            W = self.coef_              # (C, d)
            b = self.intercept_         # (C,)

            # scores: (n, C)
            scores = X_arr @ W.T
            if self.fit_intercept:
                scores = scores + b[None, :]

            # for binary case, we can return 1D array
            if scores.shape[1] == 1:
                scores = scores.reshape(-1)

            return Tensor(scores.astype(DTYPE, copy=False), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            scores = self.decision_function(X)
            if scores.ndim == 1:
                # binary: scores shape (n,)
                # class index: 0 if score <= 0, 1 if score > 0
                scores_arr = scores.data
                idx = (scores_arr > 0).astype("int64")
            else:
                scores_arr = scores.data
                idx = scores_arr.argmax(axis=1).astype("int64")

            labels = self.classes_[idx]
            return Tensor(labels, dtype=DTYPE)