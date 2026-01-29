import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClusterMixin
from LunarLearn.core import Tensor
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class KMeans(Estimator, ClusterMixin):
    """
    K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int
        Maximum number of iterations for a single run.
    tol : float
        Relative tolerance on center movement to declare convergence.
    n_init : int
        Number of different initializations and keep the best (by inertia).
    init : str
        'k-means++' or 'random'.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        init: str = "k-means++",
    ):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.init = init

        self.cluster_centers_: xp.ndarray | None = None  # (k, d)
        self.labels_: xp.ndarray | None = None           # (n,)
        self.inertia_: float | None = None               # sum of squared distances

    def _init_centers_random(self, X_arr: xp.ndarray) -> xp.ndarray:
        n_samples = X_arr.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError("n_samples < n_clusters in KMeans.")
        idx = xp.random.choice(n_samples, size=self.n_clusters, replace=False)
        return X_arr[idx].copy()

    def _init_centers_kmeanspp(self, X_arr: xp.ndarray) -> xp.ndarray:
        n_samples, n_features = X_arr.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_samples < n_clusters in KMeans.")

        centers = xp.empty((self.n_clusters, n_features), dtype=DTYPE)

        # pick first center uniformly
        first_idx = int(xp.random.randint(0, n_samples))
        centers[0] = X_arr[first_idx]

        # squared distances to nearest center
        # start with distance to first center
        diff = X_arr - centers[0][None, :]
        dist_sq = (diff * diff).sum(axis=1)  # (n,)

        for c in range(1, self.n_clusters):
            # probabilities proportional to D^2
            dist_sum = float(dist_sq.sum())
            if dist_sum <= 0.0:
                probs = xp.full(n_samples, 1.0 / n_samples, dtype=DTYPE)
            else:
                probs = dist_sq / xp.maximum(dist_sq.sum(), 1e-12)
            next_idx = int(xp.random.choice(n_samples, p=probs))
            centers[c] = X_arr[next_idx]

            # update distances to nearest center
            diff = X_arr - centers[c][None, :]
            new_dist_sq = (diff * diff).sum(axis=1)
            dist_sq = xp.minimum(dist_sq, new_dist_sq)

        return centers

    def _compute_dist_sq(self, X_arr: xp.ndarray, centers: xp.ndarray) -> xp.ndarray:
        """
        Compute squared Euclidean distances between all samples and centers.

        X_arr: (n, d)
        centers: (k, d)
        Returns:
            dist_sq: (n, k)
        """
        # ||x||^2
        x_norm = (X_arr ** 2).sum(axis=1, keepdims=True)          # (n, 1)
        # ||c||^2
        c_norm = (centers ** 2).sum(axis=1, keepdims=True).T      # (1, k)
        # X C^T
        cross = X_arr @ centers.T                                 # (n, k)
        dist_sq = x_norm + c_norm - 2.0 * cross
        return dist_sq

    def _run_kmeans_single(self, X_arr: xp.ndarray):
        """
        Run a single k-means init and return (centers, labels, inertia).
        """
        n_samples, n_features = X_arr.shape

        if self.init == "k-means++":
            centers = self._init_centers_kmeanspp(X_arr)
        elif self.init == "random":
            centers = self._init_centers_random(X_arr)
        else:
            raise ValueError(f"Unsupported init: {self.init}")

        for it in range(self.max_iter):
            dist_sq = self._compute_dist_sq(X_arr, centers)  # (n, k)
            labels = dist_sq.argmin(axis=1).astype("int64")

            # compute new centers
            new_centers = xp.zeros_like(centers)
            counts = xp.zeros((self.n_clusters,), dtype=DTYPE)

            for k in range(self.n_clusters):
                mask = labels == k
                if not xp.any(mask):
                    # empty cluster: reinitialize to random point
                    rand_idx = int(xp.random.randint(0, n_samples))
                    new_centers[k] = X_arr[rand_idx]
                    counts[k] = 1.0
                else:
                    Xk = X_arr[mask]
                    new_centers[k] = Xk.mean(axis=0)
                    counts[k] = Xk.shape[0]

            # check center shift
            shift = xp.sqrt(((centers - new_centers) ** 2).sum(axis=1)).max()
            centers = new_centers

            if shift <= self.tol:
                break

        # final inertia
        dist_sq = self._compute_dist_sq(X_arr, centers)
        labels = dist_sq.argmin(axis=1).astype("int64")
        inertia = float(dist_sq[xp.arange(n_samples), labels].sum())
        return centers, labels, inertia

    def fit(self, X: Tensor):
        with backend.no_grad():
            X = ensure_tensor(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]
            if n_samples == 0:
                raise ValueError("Cannot fit KMeans on empty data.")

            best_inertia = None
            best_centers = None
            best_labels = None

            n_init = max(self.n_init, 1)

            for _ in range(n_init):
                centers, labels, inertia = self._run_kmeans_single(X_arr)
                if best_inertia is None or inertia < best_inertia:
                    best_inertia = inertia
                    best_centers = centers
                    best_labels = labels

            self.cluster_centers_ = best_centers
            self.labels_ = best_labels
            self.inertia_ = best_inertia

        return self

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.cluster_centers_ is None:
                raise RuntimeError("KMeans not fitted.")
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            dist_sq = self._compute_dist_sq(X_arr, self.cluster_centers_)
            labels = dist_sq.argmin(axis=1).astype("int64")
            return Tensor(labels, dtype=DTYPE)
