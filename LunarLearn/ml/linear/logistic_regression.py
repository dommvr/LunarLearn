import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.core import Tensor, Parameter, ops
from LunarLearn.core.tensor import ensure_tensor
from LunarLearn.amp import amp
from LunarLearn.nn.optim.optimizers import BaseOptimizer, SGD

xp = backend.xp
DTYPE = backend.DTYPE


class LogisticRegression(Estimator, ClassifierMixin):
    def __init__(self,
                 lr: float = 0.1,
                 optim: BaseOptimizer = SGD,
                 max_iter: int = 1000,
                 C: float = 1.0,
                 fit_intercept: bool = True):
        self.optimizer = optim(lr)
        self.max_iter = max_iter
        self.C = C
        self.fit_intercept = fit_intercept
        self.W: Parameter | None = None
        self.b: Parameter | None = None

    def parameters(self, **kwargs):
        return [p for p in (self.W, self.b) if p is not None]

    def fit(self, X: Tensor, y: Tensor, use_amp: bool = True):
        X = ensure_tensor(X)
        y = ensure_tensor(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1:
            y = y.reshape(-1)
        with amp.autocast(enabled=use_amp):
            n_samples, n_features = X.shape
            n_classes = int(y.max().item() + 1)

            W = xp.zeros((n_features, n_classes), dtype=DTYPE)
            self.W = Parameter(W)
            b = xp.zeros((n_classes,), dtype=DTYPE)
            if self.fit_intercept:
                self.b = Parameter(b)
            else:
                self.b = Parameter(b, requires_grad=False)

            l2 = 1.0 / self.C if self.C > 0 else 0.0

            for _ in range(self.max_iter):
                self.optimizer.zero_grad(self.parameters())
                W = self.W.to_compute()
                b = self.b.to_compute()

                preds = ops.matmul(X, W) + b
                loss = ops.cross_entropy(preds, y) + l2 * (W ** 2).sum()
                loss = amp.scale_loss(loss)

                loss.backward()
                amp.step_if_ready(self.optimizer, self)
            
            return self
        
    def predict_proba(self, X: Tensor, use_amp: bool = True) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            with amp.autocast(enabled=use_amp):
                Wc = self.W.to_compute()
                bc = self.b.to_compute()

                logits = ops.matmul(X, Wc) + bc  # (n_samples, n_classes)
                probs = ops.softmax(logits, axis=1)
                return probs

    def predict(self, X: Tensor, use_amp: bool = True) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            with amp.autocast(enabled=use_amp):
                probs = self.predict_proba(X, use_amp=use_amp)
                return ops.argmax(probs, axis=1)