from __future__ import annotations

import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from .utils import row_scatter_add
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class BiasedMF(Estimator, RegressorMixin):
    """
    Biased Matrix Factorization for explicit feedback.

    Model:
        r_ui ≈ μ + b_u + b_i + <p_u, q_i>

    where:
        - μ is global mean
        - b_u is user bias
        - b_i is item bias
        - p_u, q_i are latent factors as in MatrixFactorization
    """

    def __init__(
        self,
        n_users: int | None = None,
        n_items: int | None = None,
        n_factors: int = 32,
        lr: float = 0.01,
        reg_factors: float = 0.0,
        reg_bias: float = 0.0,
        n_epochs: int = 10,
        shuffle: bool = True,
        init_std: float = 0.01,
        min_rating: float | None = None,
        max_rating: float | None = None,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.reg_factors = float(reg_factors)
        self.reg_bias = float(reg_bias)
        self.n_epochs = int(n_epochs)
        self.shuffle = bool(shuffle)
        self.init_std = float(init_std)
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_factors: xp.ndarray | None = None  # (n_users, k)
        self.item_factors: xp.ndarray | None = None  # (n_items, k)
        self.user_bias: xp.ndarray | None = None     # (n_users,)
        self.item_bias: xp.ndarray | None = None     # (n_items,)
        self.global_mean: float | None = None

    def _init_params(self, n_users: int, n_items: int, global_mean: float):
        rng = xp.random
        self.user_factors = (
            rng.normal(0.0, self.init_std, size=(n_users, self.n_factors))
            .astype(DTYPE)
        )
        self.item_factors = (
            rng.normal(0.0, self.init_std, size=(n_items, self.n_factors))
            .astype(DTYPE)
        )
        self.user_bias = xp.zeros((n_users,), dtype=DTYPE)
        self.item_bias = xp.zeros((n_items,), dtype=DTYPE)
        self.global_mean = float(global_mean)

    def fit(self, X: Tensor, y: Tensor):
        """
        Fit the biased MF model.

        Parameters
        ----------
        X : Tensor of shape (n_samples, 2)
            Each row: [user_index, item_index].
        y : Tensor of shape (n_samples,)
            Ratings.
        """
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 2)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data
            y_arr = y.data.astype(DTYPE, copy=False)

            if X_arr.shape[1] != 2:
                raise ValueError("X must have shape (n_samples, 2): [user_idx, item_idx].")

            user_ids = X_arr[:, 0].astype("int64")
            item_ids = X_arr[:, 1].astype("int64")

            n_samples = X_arr.shape[0]
            if n_samples == 0:
                raise ValueError("Cannot fit BiasedMF on empty data.")

            # infer n_users / n_items if not given
            n_users = self.n_users if self.n_users is not None else int(user_ids.max()) + 1
            n_items = self.n_items if self.n_items is not None else int(item_ids.max()) + 1

            if n_users <= 0 or n_items <= 0:
                raise ValueError("n_users and n_items must be positive.")

            self.n_users = n_users
            self.n_items = n_items

            # global mean
            global_mean = float(y_arr.mean())

            # init params if needed
            if any(x is None for x in (self.user_factors, self.item_factors,
                                       self.user_bias, self.item_bias,
                                       self.global_mean)):
                self._init_params(n_users, n_items, global_mean)
            else:
                if self.user_factors.shape != (n_users, self.n_factors):
                    raise ValueError("Existing user_factors shape mismatch.")
                if self.item_factors.shape != (n_items, self.n_factors):
                    raise ValueError("Existing item_factors shape mismatch.")
                if self.user_bias.shape != (n_users,):
                    raise ValueError("Existing user_bias shape mismatch.")
                if self.item_bias.shape != (n_items,):
                    raise ValueError("Existing item_bias shape mismatch.")

            P = self.user_factors   # (n_users, k)
            Q = self.item_factors   # (n_items, k)
            bu = self.user_bias     # (n_users,)
            bi = self.item_bias     # (n_items,)
            mu = self.global_mean   # scalar

            lr = self.lr
            reg_f = self.reg_factors
            reg_b = self.reg_bias

            for _ in range(self.n_epochs):
                if self.shuffle:
                    perm = xp.random.permutation(n_samples)
                    u_idx = user_ids[perm]
                    i_idx = item_ids[perm]
                    r = y_arr[perm]
                else:
                    u_idx = user_ids
                    i_idx = item_ids
                    r = y_arr

                # gather params for batch
                Pu = P[u_idx]          # (n_samples, k)
                Qi = Q[i_idx]          # (n_samples, k)
                bui = bu[u_idx]        # (n_samples,)
                bii = bi[i_idx]        # (n_samples,)

                # predictions & error
                preds = mu + bui + bii + (Pu * Qi).sum(axis=1)   # (n_samples,)
                err = r - preds

                # gradients
                # bias grads:
                # dL/db_u = -err + reg_b * b_u   => update += lr * (err - reg_b * b_u)
                # dL/db_i = -err + reg_b * b_i   => same
                grad_bu = err - reg_b * bui         # (n_samples,)
                grad_bi = err - reg_b * bii         # (n_samples,)

                # factor grads:
                # dL/dP[u] = -err * qi + reg_f * pu => update += lr * (err * qi - reg_f * pu)
                # dL/dQ[i] = -err * pu + reg_f * qi => update += lr * (err * pu - reg_f * qi)
                grad_P = err[:, None] * Qi - reg_f * Pu    # (n_samples, k)
                grad_Q = err[:, None] * Pu - reg_f * Qi    # (n_samples, k)

                # scatter-add updates
                row_scatter_add(bu, u_idx, lr * grad_bu)
                row_scatter_add(bi, i_idx, lr * grad_bi)
                row_scatter_add(P,  u_idx, lr * grad_P)
                row_scatter_add(Q,  i_idx, lr * grad_Q)

            self.user_factors = P
            self.item_factors = Q
            self.user_bias = bu
            self.item_bias = bi
            self.global_mean = mu

        return self

    def _predict_raw(self, user_ids: xp.ndarray, item_ids: xp.ndarray) -> xp.ndarray:
        if any(x is None for x in (self.user_factors, self.item_factors, self.user_bias, self.item_bias, self.global_mean)):
            raise RuntimeError("BiasedMF not fitted.")

        if user_ids.max() >= self.n_users or item_ids.max() >= self.n_items:
            raise ValueError("User or item index out of bounds for fitted model.")

        pu = self.user_factors[user_ids]    # (n, k)
        qi = self.item_factors[item_ids]    # (n, k)
        bu = self.user_bias[user_ids]       # (n,)
        bi = self.item_bias[item_ids]       # (n,)
        mu = self.global_mean

        preds = mu + bu + bi + (pu * qi).sum(axis=1)
        return preds

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict ratings for given user-item pairs.

        Parameters
        ----------
        X : Tensor of shape (n_samples, 2)

        Returns
        -------
        Tensor of shape (n_samples,)
        """
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 2)

            X_arr = X.data
            if X_arr.shape[1] != 2:
                raise ValueError("X must have shape (n_samples, 2): [user_idx, item_idx].")

            user_ids = X_arr[:, 0].astype("int64")
            item_ids = X_arr[:, 1].astype("int64")

            preds = self._predict_raw(user_ids, item_ids)

            if self.min_rating is not None:
                preds = xp.maximum(preds, self.min_rating)
            if self.max_rating is not None:
                preds = xp.minimum(preds, self.max_rating)

            return Tensor(preds.astype(DTYPE, copy=False), dtype=DTYPE)
