import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, ClassifierMixin
from LunarLearn.core import Tensor, ops
from LunarLearn.core.tensor import ensure_tensor
from LunarLearn.ml.tree.utils import _TreeNode, _encode_labels

xp = backend.xp
DTYPE = backend.DTYPE


class DecisionTreeClassifier(Estimator, ClassifierMixin):
    """
    Simple CART-style decision tree classifier (Gini impurity, binary splits).

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
        self.classes_ = None  # xp.ndarray shape (C,)
        self.n_classes_ = None

    # --- impurity & split helpers ---

    def _gini_from_counts(self, counts: xp.ndarray) -> float:
        total = counts.sum()
        if total <= 0:
            return 0.0
        p = counts / total
        return float(1.0 - (p * p).sum())

    def _find_best_split_feature_cls(
        self,
        X_col: xp.ndarray,   # shape (n,)
        y_enc: xp.ndarray,   # shape (n,), int64
        n_classes: int,
    ):
        """
        Best split on a single feature column for classification (Gini), vectorized.

        Returns:
            best_impurity, best_threshold
            If no valid split, returns (None, None).
        """
        n_samples = X_col.shape[0]
        if n_samples <= 1:
            return None, None

        # sort by feature value
        order = xp.argsort(X_col)
        X_sorted = X_col[order]
        y_sorted = y_enc[order]

        # One-hot labels: (n, C)
        Y = xp.zeros((n_samples, n_classes), dtype=DTYPE)
        Y[xp.arange(n_samples), y_sorted] = 1.0

        # prefix class counts: prefix_counts[i] = sum_{t<=i} 1_{y_t == c}
        prefix_counts = xp.cumsum(Y, axis=0)  # (n, C)

        # For a split after index i (0..n-2):
        # - left:  indices [0..i]
        # - right: indices [i+1..n-1]
        # So we use all i = 0..n-2:
        left_counts_all = prefix_counts[:-1]             # (n-1, C)
        total_counts = prefix_counts[-1]                 # (C,)

        # n_left[i] = i+1, n_right[i] = n - (i+1)
        n_left = xp.arange(1, n_samples, dtype=DTYPE)    # (n-1,)
        n_right = n_samples - n_left                     # (n-1,)

        # min_samples_leaf constraint
        valid = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)

        # ignore splits where feature value doesn't change
        diff = X_sorted[1:] != X_sorted[:-1]            # (n-1,)
        valid = valid & diff

        if not xp.any(valid):
            return None, None

        # compute right counts
        right_counts_all = total_counts[None, :] - left_counts_all  # (n-1, C)

        # Gini(left) = 1 - sum_c (p_c^2), where p_c = count_c / n_left
        # shape gymnastics: (n-1, C) / (n-1, 1)
        n_left_col = n_left[:, None]   # (n-1, 1)
        n_right_col = n_right[:, None]

        p_left = left_counts_all / xp.maximum(n_left_col, 1.0)
        p_right = right_counts_all / xp.maximum(n_right_col, 1.0)

        gini_left = 1.0 - (p_left * p_left).sum(axis=1)      # (n-1,)
        gini_right = 1.0 - (p_right * p_right).sum(axis=1)   # (n-1,)

        # weighted impurity for each split
        impurity_all = (n_left * gini_left + n_right * gini_right) / float(n_samples)

        # invalidate bad splits by setting impurity to +inf
        bad = ~valid
        if xp.any(bad):
            impurity_all = impurity_all.copy()
            impurity_all[bad] = xp.inf

        # best split index
        best_idx = int(xp.argmin(impurity_all))
        if not valid[best_idx]:
            return None, None

        # threshold between X_sorted[best_idx] and X_sorted[best_idx+1]
        thr = 0.5 * (float(X_sorted[best_idx]) + float(X_sorted[best_idx + 1]))
        best_impurity = float(impurity_all[best_idx])

        return best_impurity, thr

    def _find_best_split_cls(
        self,
        X_arr: xp.ndarray,  # (n, d)
        y_enc: xp.ndarray,  # (n,)
        n_classes: int,
    ):
        """
        Search over all features for the best split.

        Returns:
            best_feature, best_threshold, best_impurity
            If no valid split found: (None, None, None).
        """
        n_samples, n_features = X_arr.shape
        if n_samples < self.min_samples_split:
            return None, None, None

        # current node impurity
        counts = xp.bincount(y_enc, minlength=n_classes).astype(DTYPE)
        parent_impurity = self._gini_from_counts(counts)

        best_feature = None
        best_threshold = None
        best_impurity = None

        for feature_idx in range(n_features):
            col = X_arr[:, feature_idx]
            impurity, thr = self._find_best_split_feature_cls(col, y_enc, n_classes)
            if thr is None:
                continue

            if (best_impurity is None) or (impurity < best_impurity):
                best_impurity = impurity
                best_threshold = thr
                best_feature = feature_idx

        # if no split or no improvement, return None
        if best_feature is None:
            return None, None, None
        if best_impurity is not None and best_impurity >= parent_impurity:
            # no gain
            return None, None, None

        return best_feature, best_threshold, best_impurity

    # --- tree building ---

    def _build_tree_cls(
        self,
        X_arr: xp.ndarray,
        y_enc: xp.ndarray,
        depth: int,
        n_classes: int,
    ) -> _TreeNode:
        n_samples, n_features = X_arr.shape

        # class counts & leaf value
        counts = xp.bincount(y_enc, minlength=n_classes).astype(DTYPE)
        total = counts.sum()
        if total <= 0:
            # degenerate, return leaf with uniform probs
            value = xp.ones((n_classes,), dtype=DTYPE) / float(n_classes)
            return _TreeNode(is_leaf=True, value=value)

        probs = counts / total
        prediction = probs  # store full distribution

        # stopping conditions
        depth_limit = (self.max_depth is not None) and (depth >= self.max_depth)
        pure = (counts.max() == total)
        too_small = n_samples < self.min_samples_split

        if depth_limit or pure or too_small:
            return _TreeNode(is_leaf=True, value=prediction)

        # try to split
        feat, thr, best_impurity = self._find_best_split_cls(X_arr, y_enc, n_classes)
        if feat is None:
            # no useful split
            return _TreeNode(is_leaf=True, value=prediction)

        # partition data
        mask_left = X_arr[:, feat] <= thr
        mask_right = ~mask_left

        X_left, y_left = X_arr[mask_left], y_enc[mask_left]
        X_right, y_right = X_arr[mask_right], y_enc[mask_right]

        # guard against empty child (shouldn't happen if split logic is correct, but be safe)
        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            return _TreeNode(is_leaf=True, value=prediction)

        left_node = self._build_tree_cls(X_left, y_left, depth + 1, n_classes)
        right_node = self._build_tree_cls(X_right, y_right, depth + 1, n_classes)

        return _TreeNode(
            is_leaf=False,
            feature=int(feat),
            threshold=float(thr),
            left=left_node,
            right=right_node,
            value=prediction,  # parent distribution, not strictly needed for prediction
        )

    # --- public API ---

    def fit(self, X: Tensor, y: Tensor):
        with backend.no_grad():
            X = ensure_tensor(X)
            y = ensure_tensor(y)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if y.ndim > 1:
                y = y.reshape(-1)

            X_arr = X.data.astype(DTYPE, copy=False)
            classes, y_enc = _encode_labels(y)
            self.classes_ = classes
            self.n_classes_ = classes.shape[0]

            self.root = self._build_tree_cls(X_arr, y_enc, depth=0, n_classes=self.n_classes_)

        return self

    def _predict_proba_sample(self, x: xp.ndarray) -> xp.ndarray:
        """
        Traverse the tree for a single sample x (1D xp array).
        Returns xp.ndarray of shape (C,) with class probabilities.
        """
        node = self.root
        while node is not None and not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        # fallback: if somehow node is None, use uniform
        if node is None or node.value is None:
            return xp.ones((self.n_classes_,), dtype=DTYPE) / float(self.n_classes_)
        return node.value

    def predict_proba(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            if self.root is None or self.classes_ is None:
                raise RuntimeError("DecisionTreeClassifier not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples = X_arr.shape[0]
            n_classes = self.n_classes_

            probs = xp.zeros((n_samples, n_classes), dtype=DTYPE)
            for i in range(n_samples):
                probs[i] = self._predict_proba_sample(X_arr[i])

            return Tensor(probs, dtype=DTYPE)

    def predict(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            X = ensure_tensor(X)
            probs = self.predict_proba(X)
            enc_idx = ops.argmax(probs, axis=1)              # encoded indices (Tensor)
            enc_idx_arr = enc_idx.data.astype("int64")       # xp array
            labels = self.classes_[enc_idx_arr]              # original labels
            return Tensor(labels, dtype=DTYPE)