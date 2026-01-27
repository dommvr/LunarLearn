from __future__ import annotations

from typing import Iterable, Iterator, Tuple, Optional

import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor

xp = backend.xp


def _to_xp_array(y):
    if isinstance(y, Tensor):
        return y.data
    return xp.asarray(y)


def _check_n_splits(n_splits: int):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")


def _check_test_size(test_size: Optional[float | int], n_samples: int) -> int:
    if test_size is None:
        # default: 25% test
        test_size = 0.25

    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("test_size as float must be in (0, 1).")
        n_test = int(round(n_samples * test_size))
    else:
        n_test = int(test_size)

    if n_test <= 0 or n_test >= n_samples:
        raise ValueError("test_size must be in (0, n_samples).")

    return n_test


def train_test_split_indices(
    n_samples: int,
    test_size: float | int | None = None,
    shuffle: bool = True,
    stratify: Optional[Tensor | xp.ndarray | Iterable] = None,
    random_state: Optional[int] = None,
) -> Tuple[xp.ndarray, xp.ndarray]:
    """
    Return train/test indices for splitting a dataset of size n_samples.

    This is index-only; you will plug these into your dataloaders.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    test_size : float or int or None
        If float: fraction of samples for test.
        If int: number of test samples.
        If None: default 0.25.
    shuffle : bool
        Whether to shuffle before splitting.
    stratify : array-like or Tensor or None
        If given, perform stratified split based on these labels.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    train_idx : xp.ndarray, shape (n_train,)
    test_idx : xp.ndarray, shape (n_test,)
    """
    if n_samples <= 1:
        raise ValueError("n_samples must be > 1 for train/test split.")

    if random_state is not None:
        xp.random.seed(int(random_state))

    indices = xp.arange(n_samples, dtype="int64")

    if stratify is None:
        # simple split
        if shuffle:
            xp.random.shuffle(indices)

        n_test = _check_test_size(test_size, n_samples)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        return train_idx, test_idx

    # stratified split
    y = _to_xp_array(stratify)
    if y.shape[0] != n_samples:
        raise ValueError("stratify labels length must match n_samples.")

    n_test = _check_test_size(test_size, n_samples)

    # group indices by class
    unique_classes = xp.unique(y)
    class_indices = []
    for c in unique_classes:
        mask = (y == c)
        idx_c = indices[mask]
        if shuffle:
            xp.random.shuffle(idx_c)
        class_indices.append(idx_c)

    # compute desired number per class
    test_idx_list = []
    for idx_c in class_indices:
        n_c = idx_c.shape[0]
        n_c_test = int(round(n_c * (n_test / float(n_samples))))
        n_c_test = min(max(n_c_test, 1), n_c)  # at least 1, at most all
        test_idx_list.append(idx_c[:n_c_test])

    test_idx = xp.concatenate(test_idx_list, axis=0)
    if shuffle:
        xp.random.shuffle(test_idx)

    # train = all except test
    mask = xp.ones(n_samples, dtype=bool)
    mask[test_idx] = False
    train_idx = indices[mask]

    return train_idx, test_idx


class KFold:
    """
    K-fold cross-validator.

    Splits indices 0..n_samples-1 into K roughly equal folds.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle indices before splitting.
    random_state : int or None
        Seed for reproducibility (used only if shuffle=True).
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        _check_n_splits(n_splits)
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(self, n_samples: int) -> Iterator[Tuple[xp.ndarray, xp.ndarray]]:
        """
        Yield (train_idx, val_idx) for each fold.
        """
        if n_samples < self.n_splits:
            raise ValueError("n_samples must be >= n_splits.")

        indices = xp.arange(n_samples, dtype="int64")

        if self.shuffle:
            if self.random_state is not None:
                xp.random.seed(int(self.random_state))
            xp.random.shuffle(indices)

        fold_sizes = xp.full(self.n_splits, n_samples // self.n_splits, dtype="int64")
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + int(fold_size)
            val_idx = indices[start:stop]
            train_idx = xp.concatenate([indices[:start], indices[stop:]], axis=0)
            current = stop
            yield train_idx, val_idx


class StratifiedKFold:
    """
    Stratified K-Fold cross-validator.

    Preserves label proportions in each fold.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle within each class before assigning to folds.
    random_state : int or None
        Seed for reproducibility (used only if shuffle=True).
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[int] = None):
        _check_n_splits(n_splits)
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

    def split(self, n_samples: int, y: Tensor | xp.ndarray | Iterable) -> Iterator[Tuple[xp.ndarray, xp.ndarray]]:
        """
        Yield (train_idx, val_idx) for each stratified fold.

        Parameters
        ----------
        n_samples : int
        y : labels array-like of shape (n_samples,)
        """
        y_arr = _to_xp_array(y)
        if y_arr.shape[0] != n_samples:
            raise ValueError("Length of y must match n_samples.")

        indices = xp.arange(n_samples, dtype="int64")

        if self.shuffle and self.random_state is not None:
            xp.random.seed(int(self.random_state))

        # group indices by class
        unique_classes = xp.unique(y_arr)
        per_class_indices = []
        for c in unique_classes:
            mask = (y_arr == c)
            idx_c = indices[mask]
            if self.shuffle:
                xp.random.shuffle(idx_c)
            per_class_indices.append(idx_c)

        # assign per-class indices to folds in round-robin
        folds = [[] for _ in range(self.n_splits)]

        for idx_c in per_class_indices:
            n_c = idx_c.shape[0]
            # split indices of this class into n_splits groups as evenly as possible
            fold_sizes = xp.full(self.n_splits, n_c // self.n_splits, dtype="int64")
            fold_sizes[: n_c % self.n_splits] += 1

            start = 0
            for k, size_k in enumerate(fold_sizes):
                size_k = int(size_k)
                if size_k == 0:
                    continue
                stop = start + size_k
                folds[k].append(idx_c[start:stop])
                start = stop

        # build final folds
        indices_all = xp.arange(n_samples, dtype="int64")
        for k in range(self.n_splits):
            if len(folds[k]) == 0:
                val_idx = xp.empty((0,), dtype="int64")
            else:
                val_idx = xp.concatenate(folds[k], axis=0)

            # train = all except val
            mask = xp.ones(n_samples, dtype=bool)
            mask[val_idx] = False
            train_idx = indices_all[mask]

            yield train_idx, val_idx


class ShuffleSplit:
    """
    Random train/test splits for cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of re-shuffling & splitting iterations.
    test_size : float or int or None
        Fraction or number of test samples. Default 0.25 fraction.
    train_size : float or int or None
        Optional train size. If None, it is inferred as n_samples - test_size.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float | int | None = None,
        train_size: float | int | None = None,
        random_state: Optional[int] = None,
    ):
        if n_splits <= 0:
            raise ValueError("n_splits must be positive.")
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def split(self, n_samples: int) -> Iterator[Tuple[xp.ndarray, xp.ndarray]]:
        if n_samples <= 1:
            raise ValueError("n_samples must be > 1.")

        for split_idx in range(self.n_splits):
            if self.random_state is not None:
                xp.random.seed(int(self.random_state) + split_idx)

            indices = xp.random.permutation(n_samples)
            # determine test size
            if self.test_size is None:
                n_test = max(1, int(round(0.25 * n_samples)))
            else:
                n_test = _check_test_size(self.test_size, n_samples)

            if self.train_size is None:
                n_train = n_samples - n_test
            else:
                if isinstance(self.train_size, float):
                    if not (0.0 < self.train_size < 1.0):
                        raise ValueError("train_size as float must be in (0, 1).")
                    n_train = int(round(n_samples * self.train_size))
                else:
                    n_train = int(self.train_size)
                if n_train <= 0 or n_train + n_test > n_samples:
                    raise ValueError("Invalid train_size / test_size combination.")

            test_idx = indices[:n_test]
            train_idx = indices[n_test:n_test + n_train]
            yield train_idx, test_idx


class StratifiedShuffleSplit:
    """
    Stratified random train/test splits for cross-validation.

    Parameters
    ----------
    n_splits : int
        Number of re-shuffling & splitting iterations.
    test_size : float or int or None
        Fraction or number of test samples. Default 0.25 fraction.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float | int | None = None,
        random_state: Optional[int] = None,
    ):
        if n_splits <= 0:
            raise ValueError("n_splits must be positive.")
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.random_state = random_state

    def split(self, n_samples: int, y: Tensor | xp.ndarray | Iterable) -> Iterator[Tuple[xp.ndarray, xp.ndarray]]:
        y_arr = _to_xp_array(y)
        if y_arr.shape[0] != n_samples:
            raise ValueError("Length of y must match n_samples.")

        for split_idx in range(self.n_splits):
            if self.random_state is not None:
                xp.random.seed(int(self.random_state) + split_idx)

            train_idx, test_idx = train_test_split_indices(
                n_samples=n_samples,
                test_size=self.test_size,
                shuffle=True,
                stratify=y_arr,
                random_state=None,  # we already seeded
            )
            yield train_idx, test_idx