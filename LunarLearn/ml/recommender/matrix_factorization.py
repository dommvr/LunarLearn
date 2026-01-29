from __future__ import annotations

import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from .utils import row_scatter_add
from LunarLearn.core import Tensor
from LunarLearn.core.tensor import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE


class MatrixFactorization(Estimator, RegressorMixin):
    """
    Basic Matrix Factorization for explicit feedback.

    Model:
        r_ui ≈ <p_u, q_i>

    where:
        - p_u ∈ R^k is user u latent vector
        - q_i ∈ R^k is item i latent vector

    Trained with SGD on observed (user, item, rating) triples.
    """

    def __init__(
        self,
        n_users: int | None = None,
        n_items: int | None = None,
        n_factors: int = 32,
        lr: float = 0.01,
        reg: float = 0.0,
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
        self.reg = float(reg)
        self.n_epochs = int(n_epochs)
        self.shuffle = bool(shuffle)
        self.init_std = float(init_std)
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.user_factors: xp.ndarray | None = None  # (n_users, k)
        self.item_factors: xp.ndarray | None = None  # (n_items, k)

    def _init_factors(self, n_users: int, n_items: int):
        rng = xp.random
        self.user_factors = (
            rng.normal(0.0, self.init_std, size=(n_users, self.n_factors))
            .astype(DTYPE)
        )
        self.item_factors = (
            rng.normal(0.0, self.init_std, size=(n_items, self.n_factors))
            .astype(DTYPE)
        )

    def fit(self, X: Tensor, y: Tensor):
        """
        Fit the MF model.

        Parameters
        ----------
        X : Tensor of shape (n_samples, 2)
            Each row: [user_index, item_index].
        y : Tensor of shape (n_samples,)
            Ratings.
        """
        with backend.no_grad():
            X = ensure_tensor(X)
            y = ensure_tensor(y)
            # Normalize shapes
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
                raise ValueError("Cannot fit MatrixFactorization on empty data.")

            # Infer n_users / n_items if not given
            n_users = self.n_users if self.n_users is not None else int(user_ids.max()) + 1
            n_items = self.n_items if self.n_items is not None else int(item_ids.max()) + 1

            if n_users <= 0 or n_items <= 0:
                raise ValueError("n_users and n_items must be positive.")

            self.n_users = n_users
            self.n_items = n_items

            # Initialize factors if needed
            if self.user_factors is None or self.item_factors is None:
                self._init_factors(n_users, n_items)
            else:
                if self.user_factors.shape != (n_users, self.n_factors):
                    raise ValueError("Existing user_factors shape mismatch.")
                if self.item_factors.shape != (n_items, self.n_factors):
                    raise ValueError("Existing item_factors shape mismatch.")

            P = self.user_factors   # (n_users, k)
            Q = self.item_factors   # (n_items, k)

            lr = self.lr
            reg = self.reg

            for _ in range(self.n_epochs):
                # optionally shuffle
                if self.shuffle:
                    perm = xp.random.permutation(n_samples)
                    u_idx = user_ids[perm]
                    i_idx = item_ids[perm]
                    r = y_arr[perm]
                else:
                    u_idx = user_ids
                    i_idx = item_ids
                    r = y_arr

                # gather user/item factors for all samples
                Pu = P[u_idx]           # (n_samples, k)
                Qi = Q[i_idx]           # (n_samples, k)

                # predictions & errors
                preds = (Pu * Qi).sum(axis=1)   # (n_samples,)
                err = r - preds                 # (n_samples,)

                # gradients
                # dL/dP[u] = -err * qi + reg * pu  => update += lr * (err * qi - reg * pu)
                # dL/dQ[i] = -err * pu + reg * qi  => update += lr * (err * pu - reg * qi)
                grad_P = err[:, None] * Qi - reg * Pu    # (n_samples, k)
                grad_Q = err[:, None] * Pu - reg * Qi    # (n_samples, k)

                # apply updates with scatter-add (handles repeated u/i)
                row_scatter_add(P, u_idx, lr * grad_P)
                row_scatter_add(Q, i_idx, lr * grad_Q)

            self.user_factors = P
            self.item_factors = Q

        return self

    def _predict_raw(self, user_ids: xp.ndarray, item_ids: xp.ndarray) -> xp.ndarray:
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("MatrixFactorization not fitted.")

        # bounds check
        if user_ids.max() >= self.n_users or item_ids.max() >= self.n_items:
            raise ValueError("User or item index out of bounds for fitted model.")

        pu = self.user_factors[user_ids]  # (n, k)
        qi = self.item_factors[item_ids]  # (n, k)
        preds = (pu * qi).sum(axis=1)     # (n,)
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
            X = ensure_tensor(X)
            if X.ndim == 1:
                X = X.reshape(-1, 2)

            X_arr = X.data
            if X_arr.shape[1] != 2:
                raise ValueError("X must have shape (n_samples, 2): [user_idx, item_idx].")

            user_ids = X_arr[:, 0].astype("int64")
            item_ids = X_arr[:, 1].astype("int64")

            preds = self._predict_raw(user_ids, item_ids)

            # optional clipping to rating range
            if self.min_rating is not None:
                preds = xp.maximum(preds, self.min_rating)
            if self.max_rating is not None:
                preds = xp.minimum(preds, self.max_rating)

            return Tensor(preds.astype(DTYPE, copy=False), dtype=DTYPE)
