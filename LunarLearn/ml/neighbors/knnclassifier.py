import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import ClassifierMixin
from LunarLearn.ml.neighbors import BaseKNeighbors
from LunarLearn.core import Tensor, ops
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class KNNClassifier(BaseKNeighbors, ClassifierMixin):
    """
    k-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to use.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__(n_neighbors=n_neighbors)
        self.classes_ = None

    def fit(self, X: Tensor, y: Tensor):
        """
        Store training data and build label encoding.
        """
        with backend.no_grad():
            X = ensure_tensor(X)
            y = ensure_tensor(y)
            # Normalize shapes
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            # raw labels as xp array
            y_arr = y.data

            # sorted unique classes
            self.classes_ = xp.unique(y_arr)

            # map labels -> indices 0..C-1 using searchsorted
            # works because classes_ is sorted
            y_encoded = xp.searchsorted(self.classes_, y_arr)

            # store tensors
            self.X_train = X
            self.y_train = Tensor(y_encoded, dtype=DTYPE)

        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        """
        Predict class probabilities for each sample.

        Returns
        -------
        Tensor of shape (n_samples, n_classes)
        """
        X = ensure_tensor(X)
        self._check_is_fitted()
        if self.classes_ is None:
            raise RuntimeError("KNNClassifier not fitted properly: classes_ is None.")

        idx, k = self._kneighbors_indices(X)

        y_enc_arr = self.y_train.data.astype("int64")   # encoded labels
        n_query = idx.shape[0]
        n_classes = self.classes_.shape[0]

        if n_query == 0:
            # empty input, empty output; cheap early exit
            return Tensor(xp.zeros((0, n_classes), dtype=DTYPE), dtype=DTYPE)

        # neighbors_enc: (n_query, k)
        neighbors_enc = y_enc_arr[idx]

        # Build flat indices into a virtual (n_query, n_classes) matrix
        # row indices: 0,0,...,0, 1,1,...,1, ..., n_query-1 repeated k times each
        rows = xp.repeat(xp.arange(n_query, dtype="int64"), k)       # (n_query * k,)

        # class indices for each neighbor
        cols = neighbors_enc.ravel().astype("int64")                 # (n_query * k,)

        # flat index = row * n_classes + col
        flat_idx = rows * n_classes + cols                           # (n_query * k,)

        # bincount over flattened space, then reshape back to (n_query, n_classes)
        counts_flat = xp.bincount(flat_idx,
                                minlength=n_query * n_classes).astype(DTYPE)
        probs_arr = counts_flat.reshape(n_query, n_classes) / float(k)

        return Tensor(probs_arr, dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict class labels for each sample (original label space).
        """
        with backend.no_grad():
            X = ensure_tensor(X)
            probs = self.predict_proba(X)
            enc_idx = ops.argmax(probs, axis=1)          # Tensor of encoded indices
            enc_idx_arr = enc_idx.data.astype("int64")  # xp array

            # map encoded indices -> original labels
            labels = self.classes_[enc_idx_arr]         # xp array of original labels

            return Tensor(labels, dtype=DTYPE)