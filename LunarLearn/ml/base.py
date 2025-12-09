from abc import ABC, abstractmethod
from LunarLearn.train.metrics import R2Score, Accuracy

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
        y_pred = self.predict(X)
        # Here you'd probably call LunarLearn.metrics.r2_score
        return 1.0 - ((y - y_pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12)


class ClassifierMixin:
    """Mixin adding a default score method for classifiers."""

    def score(self, X, y):
        """
        Default accuracy.
        """
        y_pred = self.predict(X)
        return (y_pred == y).astype("float32").mean()


class ClusterMixin:
    def fit_predict(self, X, y=None, **fit_params):
        if y is None:
            return self.fit(X, **fit_params).predict(X)
        else:
            return self.fit(X, y, **fit_params).predict(X)
        

class TransformMixin:
    """
    Mixin for estimators that implement transform.

    Provides:
    - fit_transform(X, **fit_params)
    """
    def fit_transform(self, X, **fit_params):
        self.fit(X, **fit_params)
        return self.transform(X)