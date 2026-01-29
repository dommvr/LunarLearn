from abc import ABC, abstractmethod
from LunarLearn.train.metrics import R2Score, Accuracy
from LunarLearn.core.tensor import ensure_tensor


class Estimator(ABC):
    """Base estimator with sklearn-like interface."""

    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class RegressorMixin:
    """Mixin adding a default score method for regressors."""

    def score(self, X, y):
        """
        Default R^2 score. Replace with your metrics if needed.
        """
        X = ensure_tensor(X)
        y = ensure_tensor(y)
        metric = R2Score()
        y_pred = self.predict(X)
        return metric(y_pred, y)


class ClassifierMixin:
    """Mixin adding a default score method for classifiers."""

    def score(self, X, y):
        """
        Default accuracy.
        """
        X = ensure_tensor(X)
        y = ensure_tensor(y)
        metric = Accuracy()
        y_pred = self.predict(X)
        return metric(y_pred, y)


class ClusterMixin:
    def fit_predict(self, X, y=None, **fit_params):
        X = ensure_tensor(X)
        if y is None:
            return self.fit(X, **fit_params).predict(X)
        else:
            y = ensure_tensor(y)
            return self.fit(X, y, **fit_params).predict(X)
        

class TransformMixin:
    """
    Mixin for estimators that implement transform.

    Provides:
    - fit_transform(X, **fit_params)
    """
    def fit_transform(self, X, **fit_params):
        X = ensure_tensor(X)
        self.fit(X, **fit_params)
        return self.transform(X)