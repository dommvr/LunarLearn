import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClusterMixin
from LunarLearn.ml.cluster import KMeans
from LunarLearn.core import Tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


class GaussianMixture(Estimator, ClusterMixin):
    """
    Gaussian Mixture Model with full covariance, fit by EM.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on log-likelihood improvement.
    reg_covar : float
        Added to diagonal of covariance matrices for stability.
    """

    def __init__(
        self,
        n_components: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
    ):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)

        self.weights_: xp.ndarray | None = None      # (k,)
        self.means_: xp.ndarray | None = None        # (k, d)
        self.covariances_: xp.ndarray | None = None  # (k, d, d)

        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float | None = None       # final log-likelihood per sample

    def _init_params(self, X_arr: xp.ndarray):
        n_samples, n_features = X_arr.shape
        k = self.n_components

        # init means with KMeans
        km = KMeans(n_clusters=k, max_iter=20, n_init=1, init="k-means++")
        km.fit(Tensor(X_arr, dtype=DTYPE))
        means = km.cluster_centers_.copy()

        # equal weights
        weights = xp.full((k,), 1.0 / k, dtype=DTYPE)

        # covariances: global empirical + reg_covar
        cov_global = xp.cov(X_arr.T).astype(DTYPE, copy=False)  # (d, d)
        if cov_global.ndim == 0:
            cov_global = cov_global.reshape(1, 1)
        covariances = xp.stack([cov_global + self.reg_covar * xp.eye(cov_global.shape[0], dtype=DTYPE)
                                for _ in range(k)], axis=0)

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    def _estimate_log_gaussian_prob(
        self,
        X_arr: xp.ndarray,
        means: xp.ndarray,
        covariances: xp.ndarray,
    ) -> xp.ndarray:
        """
        Compute log N(x | mean_k, cov_k) for all samples and components.

        X_arr: (n, d)
        means: (k, d)
        covariances: (k, d, d)
        Returns:
            log_prob: (n, k)
        """
        n_samples, n_features = X_arr.shape
        k = means.shape[0]

        log_prob = xp.empty((n_samples, k), dtype=DTYPE)

        for j in range(k):
            mean = means[j]                      # (d,)
            cov = covariances[j]                # (d, d)

            # regularize & ensure symmetric
            cov = 0.5 * (cov + cov.T) + self.reg_covar * xp.eye(n_features, dtype=DTYPE)

            # log det & inverse
            sign, logdet = xp.linalg.slogdet(cov)
            if sign <= 0:
                # fallback: add larger reg and recompute
                cov = cov + self.reg_covar * xp.eye(n_features, dtype=DTYPE)
                sign, logdet = xp.linalg.slogdet(cov)
                if sign <= 0:
                    # give up, set very low prob
                    log_prob[:, j] = -1e12
                    continue

            diff = X_arr - mean[None, :]        # (n, d)
            # solve cov * y = diff^T  => y^T = diff^T * cov^{-1}
            # but easier: solve for each sample in batch
            # use solve on transpose:
            sol = xp.linalg.solve(cov, diff.T).T    # (n, d)

            mahal = (diff * sol).sum(axis=1)        # (n,)

            log_norm = -0.5 * (n_features * math.log(2.0 * math.pi) + logdet)
            log_prob[:, j] = log_norm - 0.5 * mahal

        return log_prob

    def _estimate_log_resp(self, X_arr: xp.ndarray):
        """
        Compute log responsibilities log r_{ik} ∝ log pi_k + log N(x_i | mean_k, cov_k).

        Returns:
            log_resp: (n, k), log_prob_norm: scalar per-sample log-likelihood mean.
        """
        weights = self.weights_
        means = self.means_
        covariances = self.covariances_

        log_gauss = self._estimate_log_gaussian_prob(X_arr, means, covariances)  # (n, k)

        # log pi_k
        log_weights = xp.log(weights + 1e-12)[None, :]    # (1, k)

        # log joint = log pi_k + log N
        log_prob = log_gauss + log_weights               # (n, k)

        # logsumexp over components
        max_log = log_prob.max(axis=1, keepdims=True)    # (n, 1)
        log_sum = max_log + xp.log(
            xp.maximum(xp.exp(log_prob - max_log).sum(axis=1, keepdims=True), 1e-12)
        )                                                # (n, 1)

        log_resp = log_prob - log_sum                    # (n, k)

        # average log-likelihood per sample
        lower_bound = float(log_sum.mean())
        return log_resp, lower_bound

    def _m_step(self, X_arr: xp.ndarray, log_resp: xp.ndarray):
        """
        M-step: update weights, means, covariances using responsibilities.
        """
        n_samples, n_features = X_arr.shape
        k = self.n_components

        resp = xp.exp(log_resp)  # (n, k)
        # effective cluster weights
        Nk = resp.sum(axis=0) + 1e-12        # (k,)

        # weights
        weights = Nk / Nk.sum()

        # means
        means = (resp.T @ X_arr) / Nk[:, None]   # (k, d)

        # covariances
        covariances = xp.empty((k, n_features, n_features), dtype=DTYPE)
        for j in range(k):
            diff = X_arr - means[j][None, :]
            # weight each sample by resp[i, j]
            # cov = (sum_i r_ij * diff_i^T diff_i) / Nk[j]
            rj = resp[:, j][:, None]  # (n, 1)
            weighted = diff * xp.sqrt(rj)  # (n, d)
            cov = (weighted.T @ weighted) / Nk[j]
            # regularize
            cov = cov + self.reg_covar * xp.eye(n_features, dtype=DTYPE)
            covariances[j] = cov

        self.weights_ = weights.astype(DTYPE)
        self.means_ = means.astype(DTYPE)
        self.covariances_ = covariances

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_arr = X.data.astype(DTYPE, copy=False)

            n_samples, n_features = X_arr.shape
            if n_samples == 0:
                raise ValueError("Cannot fit GaussianMixture on empty data.")

            if self.n_components <= 0:
                raise ValueError("n_components must be positive.")

            self._init_params(X_arr)

            lower_bound_old = -xp.inf
            self.converged_ = False

            for n_iter in range(1, self.max_iter + 1):
                log_resp, lower_bound = self._estimate_log_resp(X_arr)
                self._m_step(X_arr, log_resp)

                change = lower_bound - lower_bound_old
                lower_bound_old = lower_bound

                if abs(change) < self.tol:
                    self.converged_ = True
                    self.n_iter_ = n_iter
                    break
            else:
                self.n_iter_ = self.max_iter

            self.lower_bound_ = lower_bound_old

        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.weights_ is None:
                raise RuntimeError("GaussianMixture not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_arr = X.data.astype(DTYPE, copy=False)

            log_resp, _ = self._estimate_log_resp(X_arr)
            resp = xp.exp(log_resp)
            return Tensor(resp.astype(DTYPE), dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            resp = self.predict_proba(X)
            labels = resp.data.argmax(axis=1).astype("int64")
            return Tensor(labels, dtype=DTYPE)

    def score_samples(self, X: Tensor) -> Tensor:
        """
        Return log probability density for each sample under the model.
        """
        with backend.no_grad():
            if self.weights_ is None:
                raise RuntimeError("GaussianMixture not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X_arr = X.data.astype(DTYPE, copy=False)

            weights = self.weights_
            means = self.means_
            covariances = self.covariances_

            log_gauss = self._estimate_log_gaussian_prob(X_arr, means, covariances)  # (n, k)
            log_weights = xp.log(weights + 1e-12)[None, :]
            log_prob = log_gauss + log_weights

            max_log = log_prob.max(axis=1, keepdims=True)
            log_sum = max_log + xp.log(
                xp.maximum(xp.exp(log_prob - max_log).sum(axis=1, keepdims=True), 1e-12)
            )
            return Tensor(log_sum.reshape(-1).astype(DTYPE), dtype=DTYPE)