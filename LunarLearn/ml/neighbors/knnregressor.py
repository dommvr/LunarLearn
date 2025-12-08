import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import RegressorMixin
from LunarLearn.ml.neighbors import BaseKNeighbors
from LunarLearn.core import Tensor

DTYPE = backend.DTYPE


class KNNRegressor(BaseKNeighbors, RegressorMixin):
    """
    k-Nearest Neighbors regressor.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to use.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__(n_neighbors=n_neighbors)

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict continuous targets by averaging neighbor targets.
        """
        self._check_is_fitted()
        y_train = self.y_train

        idx, k = self._kneighbors_indices(X)
        y_arr = y_train.data  # (n_train,) or (n_train, d)

        with backend.no_grad():
            # Gather neighbor targets
            # idx: (n_q, k)
            neighbors = y_arr[idx]  # (n_q, k) or (n_q, k, d)
            # Mean over neighbors axis
            preds_arr = neighbors.mean(axis=1)

            return Tensor(preds_arr, dtype=DTYPE)