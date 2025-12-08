import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator
from LunarLearn.core import Tensor, ops

xp = backend.xp


class KNeighborsBase(Estimator):
    """
    Base class for KNN models.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to use.
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = int(n_neighbors)
        self.X_train: Tensor | None = None
        self.y_train: Tensor | None = None

    def fit(self, X: Tensor, y: Tensor):
        """
        Store the training data. KNN is a lazy learner.
        """
        with backend.no_grad():
            # Normalize shapes
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            self.X_train = X
            self.y_train = y

        return self

    def _check_is_fitted(self):
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("KNN model is not fitted yet. Call `fit` first.")

    def _pairwise_distances(self, X: Tensor, eps: float = 1e-12) -> Tensor:
        """
        Compute pairwise L2 distances between query X and stored X_train.

        Returns:
            Tensor of shape (n_query, n_train)
        """
        self._check_is_fitted()
        X_train = self.X_train

        # X: (n_q, d), X_train: (n_t, d)
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Broadcast: (n_q, 1, d) - (1, n_t, d) -> (n_q, n_t, d)
            diff = X[:, None, :] - X_train[None, :, :]
            sq = diff ** 2
            # Sum over features -> (n_q, n_t)
            dist_sq = sq.sum(axis=-1)
            # L2 distance
            dists = ops.sqrt(dist_sq + eps)
            return dists

    def _kneighbors_indices(self, X: Tensor):
        """
        Return indices of k nearest neighbors for each query row in X.

        Returns:
            xp.ndarray of shape (n_q, k) with integer indices.
        """
        dists = self._pairwise_distances(X)
        dists_arr = dists.data  # xp array

        n_train = dists_arr.shape[1]
        k = min(self.n_neighbors, n_train)
        if k <= 0:
            raise ValueError("n_neighbors must be positive and <= n_train.")

        # argpartition for top-k (smallest distances)
        # partition at k-1, then take first k indices
        idx = xp.argpartition(dists_arr, kth=k - 1, axis=1)[:, :k]
        return idx, k