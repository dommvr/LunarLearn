import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, RegressorMixin
from LunarLearn.core import Tensor
from LunarLearn.ml.tree.utils import _TreeNode

xp = backend.xp
DTYPE = backend.DTYPE


class DecisionTreeRegressor(Estimator, RegressorMixin):
    """
    Simple CART-style decision tree regressor (MSE criterion, binary splits).

    Parameters
    ----------
    max_depth : int | None
        Maximum tree depth. None means unlimited.
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root: _TreeNode | None = None

    def _mse_from_stats(self, sum_y: float, sum_y2: float, n: int) -> float:
        if n <= 0:
            return 0.0
        mean = sum_y / n
        return float(sum_y2 / n - mean * mean)

    def _find_best_split_feature_reg(
        self,
        X_col: xp.ndarray,   # (n,)
        y_arr: xp.ndarray,   # (n,)
    ):
        """
        Best split on a single feature column for regression (MSE), vectorized.

        Returns:
            best_impurity, best_threshold
            If no valid split, returns (None, None).
        """
        n_samples = X_col.shape[0]
        if n_samples <= 1:
            return None, None

        # Sort by feature
        order = xp.argsort(X_col)
        X_sorted = X_col[order]
        y_sorted = y_arr[order]

        # Prefix sums over y and y^2
        cumsum_y = xp.cumsum(y_sorted)          # (n,)
        cumsum_y2 = xp.cumsum(y_sorted ** 2)    # (n,)

        total_sum_y = cumsum_y[-1]
        total_sum_y2 = cumsum_y2[-1]

        # We consider splits after index i = 0..n-2
        # For each split index j (0..n-2):
        #   left: 0..j, size n_left = j+1
        #   right: j+1..n-1, size n_right = n - (j+1)
        n_left = xp.arange(1, n_samples, dtype=DTYPE)   # (n-1,)
        n_right = n_samples - n_left                    # (n-1,)

        # min_samples_leaf constraint
        valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)

        # reject splits where feature doesn't change
        diff = X_sorted[1:] != X_sorted[:-1]           # (n-1,)
        valid = valid & diff

        if not xp.any(valid):
            return None, None

        # Left sums for each split j: sum over [0..j]
        sum_left = cumsum_y[:-1]                       # (n-1,)
        sum_left2 = cumsum_y2[:-1]                     # (n-1,)

        # Right sums for each split j: total - left
        sum_right = total_sum_y - sum_left             # (n-1,)
        sum_right2 = total_sum_y2 - sum_left2          # (n-1,)

        # MSE for left/right for all splits at once
        # mse = E[y^2] - (E[y])^2
        n_left_safe = xp.maximum(n_left, 1.0)
        n_right_safe = xp.maximum(n_right, 1.0)

        mean_left = sum_left / n_left_safe
        mean_right = sum_right / n_right_safe

        mse_left = sum_left2 / n_left_safe - mean_left ** 2      # (n-1,)
        mse_right = sum_right2 / n_right_safe - mean_right ** 2  # (n-1,)

        # Weighted impurity
        impurity_all = (n_left * mse_left + n_right * mse_right) / float(n_samples)

        # Invalidate bad splits
        bad = ~valid
        if xp.any(bad):
            impurity_all = impurity_all.copy()
            impurity_all[bad] = xp.inf

        # Best split index
        best_idx = int(xp.argmin(impurity_all))
        if not valid[best_idx]:
            return None, None

        # Threshold halfway between X_sorted[best_idx] and X_sorted[best_idx + 1]
        thr = 0.5 * (float(X_sorted[best_idx]) + float(X_sorted[best_idx + 1]))
        best_impurity = float(impurity_all[best_idx])

        return best_impurity, thr

    def _find_best_split_reg(
        self,
        X_arr: xp.ndarray,  # (n, d)
        y_arr: xp.ndarray,  # (n,)
    ):
        n_samples, n_features = X_arr.shape
        if n_samples < self.min_samples_split:
            return None, None, None

        # current MSE at node
        total_sum_y = y_arr.sum()
        total_sum_y2 = (y_arr ** 2).sum()
        parent_mse = self._mse_from_stats(total_sum_y, total_sum_y2, n_samples)

        best_feature = None
        best_threshold = None
        best_impurity = None

        for feature_idx in range(n_features):
            col = X_arr[:, feature_idx]
            impurity, thr = self._find_best_split_feature_reg(col, y_arr)
            if thr is None:
                continue

            if (best_impurity is None) or (impurity < best_impurity):
                best_impurity = impurity
                best_threshold = thr
                best_feature = feature_idx

        if best_feature is None:
            return None, None, None
        if best_impurity is not None and best_impurity >= parent_mse:
            return None, None, None

        return best_feature, best_threshold, best_impurity

    def _build_tree_reg(
        self,
        X_arr: xp.ndarray,
        y_arr: xp.ndarray,
        depth: int,
    ) -> _TreeNode:
        n_samples, n_features = X_arr.shape

        # leaf prediction: mean
        if n_samples == 0:
            return _TreeNode(is_leaf=True, value=0.0)

        mean_value = float(y_arr.mean())

        depth_limit = (self.max_depth is not None) and (depth >= self.max_depth)
        too_small = n_samples < self.min_samples_split

        if depth_limit or too_small:
            return _TreeNode(is_leaf=True, value=mean_value)

        feat, thr, best_impurity = self._find_best_split_reg(X_arr, y_arr)
        if feat is None:
            return _TreeNode(is_leaf=True, value=mean_value)

        mask_left = X_arr[:, feat] <= thr
        mask_right = ~mask_left

        X_left, y_left = X_arr[mask_left], y_arr[mask_left]
        X_right, y_right = X_arr[mask_right], y_arr[mask_right]

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            return _TreeNode(is_leaf=True, value=mean_value)

        left_node = self._build_tree_reg(X_left, y_left, depth + 1)
        right_node = self._build_tree_reg(X_right, y_right, depth + 1)

        return _TreeNode(
            is_leaf=False,
            feature=int(feat),
            threshold=float(thr),
            left=left_node,
            right=right_node,
            value=mean_value,
        )

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            y_arr = y.data.astype(DTYPE, copy=False)

            self.root = self._build_tree_reg(X_arr, y_arr, depth=0)

        return self

    def _predict_sample(self, x: xp.ndarray) -> float:
        node = self.root
        while node is not None and not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        if node is None or node.value is None:
            return 0.0
        return float(node.value)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.root is None:
                raise RuntimeError("DecisionTreeRegressor not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]

            preds = xp.zeros((n_samples,), dtype=DTYPE)
            for i in range(n_samples):
                preds[i] = self._predict_sample(X_arr[i])

            return Tensor(preds, dtype=DTYPE)