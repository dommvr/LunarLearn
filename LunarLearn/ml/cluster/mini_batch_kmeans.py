import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClusterMixin
from LunarLearn.ml.cluster import KMeans
from LunarLearn.core import Tensor
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class MiniBatchKMeans(Estimator, ClusterMixin):
    """
    Mini-Batch K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    batch_size : int
        Size of the mini-batches.
    max_iter : int
        Number of mini-batch updates.
    tol : float
        Tolerance on center movement (for optional early stopping).
    init : str
        'k-means++' or 'random'.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 100,
        max_iter: int = 100,
        tol: float = 1e-3,
        init: str = "k-means++",
    ):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.init = init

        self.cluster_centers_: xp.ndarray | None = None  # (k, d)
        self.labels_: xp.ndarray | None = None
        self.inertia_: float | None = None

        self._counts: xp.ndarray | None = None           # (k,)

    def _init_centers(self, X_arr: xp.ndarray) -> xp.ndarray:
        if self.init == "k-means++":
            # reuse KMeans logic
            km = KMeans(
                n_clusters=self.n_clusters,
                max_iter=10,
                tol=1e-4,
                n_init=1,
                init="k-means++",
            )
            # quick init run
            X_tensor = Tensor(X_arr, dtype=DTYPE)
            km.fit(X_tensor)
            return km.cluster_centers_.copy()
        elif self.init == "random":
            n_samples = X_arr.shape[0]
            if n_samples < self.n_clusters:
                raise ValueError("n_samples < n_clusters in MiniBatchKMeans.")
            idx = xp.random.choice(n_samples, size=self.n_clusters, replace=False)
            return X_arr[idx].copy()
        else:
            raise ValueError(f"Unsupported init: {self.init}")

    def _compute_dist_sq(self, X_arr: xp.ndarray, centers: xp.ndarray) -> xp.ndarray:
        x_norm = (X_arr ** 2).sum(axis=1, keepdims=True)
        c_norm = (centers ** 2).sum(axis=1, keepdims=True).T
        cross = X_arr @ centers.T
        return x_norm + c_norm - 2.0 * cross

    def fit(self, X: Tensor):
        with backend.no_grad():
            X = ensure_tensor(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_arr = X.data.astype(DTYPE, copy=False)

            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit MiniBatchKMeans on empty data.")

            centers = self._init_centers(X_arr)
            counts = xp.zeros((self.n_clusters,), dtype=DTYPE)

            batch_size = min(self.batch_size, n_samples)

            prev_centers = centers.copy()

            for _ in range(self.max_iter):
                # sample a mini-batch
                indices = xp.random.randint(0, n_samples, size=(batch_size,))
                X_batch = X_arr[indices]

                # assign batch to clusters
                dist_sq = self._compute_dist_sq(X_batch, centers)  # (b, k)
                labels_batch = dist_sq.argmin(axis=1).astype("int64")

                # accumulate stats per cluster
                # for each cluster j, sum of x in batch and count
                for j in range(self.n_clusters):
                    mask = labels_batch == j
                    if not xp.any(mask):
                        continue
                    Xj = X_batch[mask]
                    batch_count = Xj.shape[0]

                    # update counts and centers with online mean
                    counts_j_old = counts[j]
                    counts_j_new = counts_j_old + batch_count

                    # weighted mean: (old_sum + batch_sum) / new_count
                    if counts_j_old == 0:
                        centers[j] = Xj.mean(axis=0)
                    else:
                        # old_sum = centers[j] * counts_j_old
                        old_sum = centers[j] * counts_j_old
                        new_sum = old_sum + Xj.sum(axis=0)
                        centers[j] = new_sum / counts_j_new

                    counts[j] = counts_j_new

                # check center movement (optional early stopping)
                shift = xp.sqrt(((centers - prev_centers) ** 2).sum(axis=1)).max()
                prev_centers[...] = centers
                if shift <= self.tol:
                    break

            # assign full data to final centers
            full_dist_sq = self._compute_dist_sq(X_arr, centers)
            labels = full_dist_sq.argmin(axis=1).astype("int64")
            inertia = float(full_dist_sq[xp.arange(n_samples), labels].sum())

            self.cluster_centers_ = centers
            self.labels_ = labels
            self.inertia_ = inertia
            self._counts = counts

        return self

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.cluster_centers_ is None:
                raise RuntimeError("MiniBatchKMeans not fitted.")
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_arr = X.data.astype(DTYPE, copy=False)
            dist_sq = self._compute_dist_sq(X_arr, self.cluster_centers_)
            labels = dist_sq.argmin(axis=1).astype("int64")
            return Tensor(labels, dtype=DTYPE)