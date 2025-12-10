import LunarLearn.core.backend.backend as backend
from LunarLearn.ml.base import Estimator, TransformMixin
from LunarLearn.core import Tensor
import itertools

xp = backend.xp
DTYPE = backend.DTYPE


class PolynomialFeatures(Estimator, TransformMixin):
    """
    Generate polynomial and interaction features.

    Given input X of shape (n_samples, n_features), generates features:

        [1, x1, x2, ..., x_d, x1^2, x1 x2, ..., x_d^degree]

    depending on parameters.

    Parameters
    ----------
    degree : int
        Maximum degree of polynomial features.
    include_bias : bool
        If True, include a bias (all-ones) column as first feature.
    interaction_only : bool
        If True, only interaction features (no powers > 1 of a single feature).
        That is, exponents are 0 or 1, and sum(exponents) <= degree.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = True,
        interaction_only: bool = False,
    ):
        if degree < 1:
            raise ValueError("degree must be >= 1.")

        self.degree = int(degree)
        self.include_bias = bool(include_bias)
        self.interaction_only = bool(interaction_only)

        self.n_input_features_: int | None = None
        self.n_output_features_: int | None = None
        # list of tuples of exponents per output feature, length n_output_features_
        self.powers_: list[tuple[int, ...]] | None = None

    def _generate_powers(self, n_features: int):
        """
        Generate exponent tuples for each output feature.

        Returns
        -------
        powers : list of tuples (e_1, ..., e_d)
        """
        powers: list[tuple[int, ...]] = []

        # bias term
        if self.include_bias:
            powers.append((0,) * n_features)

        if self.interaction_only:
            # combinations of features with exponent 0 or 1
            # degree is max number of features interacting
            for deg in range(1, self.degree + 1):
                for comb in itertools.combinations(range(n_features), deg):
                    exp = [0] * n_features
                    for idx in comb:
                        exp[idx] = 1
                    powers.append(tuple(exp))
        else:
            # full polynomial up to given degree
            # use combinations with replacement over features to build monomials
            for deg in range(1, self.degree + 1):
                # each monomial is represented by a multiset of feature indices
                for comb in itertools.combinations_with_replacement(range(n_features), deg):
                    exp = [0] * n_features
                    for idx in comb:
                        exp[idx] += 1
                    powers.append(tuple(exp))

        return powers

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            X_arr = X.data
            _, n_features = X_arr.shape

            self.n_input_features_ = n_features
            self.powers_ = self._generate_powers(n_features)
            self.n_output_features_ = len(self.powers_)

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.powers_ is None or self.n_input_features_ is None:
                raise RuntimeError("PolynomialFeatures not fitted.")

            if X.ndim == 1:
                X = X.reshape(-1, self.n_input_features_)

            X_arr = X.data.astype(DTYPE, copy=False)
            n_samples, n_features = X_arr.shape
            if n_features != self.n_input_features_:
                raise ValueError(
                    f"Expected {self.n_input_features_} features, got {n_features}."
                )

            n_out = self.n_output_features_
            out = xp.empty((n_samples, n_out), dtype=DTYPE)

            # For each output feature, compute product over input features
            for j, exp in enumerate(self.powers_):
                exp_arr = xp.asarray(exp, dtype=DTYPE)  # (d,)
                # For speed, handle some special cases:
                if xp.all(exp_arr == 0):
                    out[:, j] = 1.0
                else:
                    # X_arr ** exp, but exp is per-feature
                    # X_pow: (n, d)
                    X_pow = xp.ones_like(X_arr)
                    # only compute where exponent > 0
                    for k in range(n_features):
                        e = exp[k]
                        if e == 0:
                            continue
                        elif e == 1:
                            X_pow[:, k] = X_arr[:, k]
                        else:
                            X_pow[:, k] = X_arr[:, k] ** e
                    out[:, j] = X_pow.prod(axis=1)

            return Tensor(out, dtype=DTYPE)


class LabelEncoder(Estimator):
    """
    Encode target labels (or a single categorical feature) as integers 0..n_classes-1.

    Parameters
    ----------
    None
    """

    def __init__(self):
        self.classes_: xp.ndarray | None = None

    def fit(self, y: Tensor):
        with backend.no_grad():
            if y.ndim > 1:
                y = y.reshape(-1)

            y_arr = y.data
            classes = xp.unique(y_arr)
            self.classes_ = classes

        return self

    def transform(self, y: Tensor) -> Tensor:
        with backend.no_grad():
            if self.classes_ is None:
                raise RuntimeError("LabelEncoder not fitted.")

            if y.ndim > 1:
                y = y.reshape(-1)

            y_arr = y.data
            # map labels -> indices in classes_ via searchsorted
            enc = xp.searchsorted(self.classes_, y_arr)
            return Tensor(enc.astype(DTYPE), dtype=DTYPE)

    def inverse_transform(self, y_enc: Tensor) -> Tensor:
        with backend.no_grad():
            if self.classes_ is None:
                raise RuntimeError("LabelEncoder not fitted.")

            if y_enc.ndim > 1:
                y_enc = y_enc.reshape(-1)

            y_enc_arr = y_enc.data.astype("int64")
            if y_enc_arr.min() < 0 or y_enc_arr.max() >= self.classes_.shape[0]:
                raise ValueError("Encoded labels out of range for LabelEncoder.")

            decoded = self.classes_[y_enc_arr]
            return Tensor(decoded, dtype=DTYPE)


class OneHotEncoder(Estimator, TransformMixin):
    """
    Simple one-hot encoder for a single categorical feature or encoded labels.

    This is intentionally minimal:
    - Input is 1D (n_samples,) or 2D with a single column (n_samples, 1).
    - It one-hot encodes that single column.

    Parameters
    ----------
    dtype : DTYPE-like
        Output dtype.
    """

    def __init__(self, dtype=None):
        self.dtype = DTYPE if dtype is None else dtype
        self.categories_: xp.ndarray | None = None   # (C,)
        self.n_categories_: int | None = None

    def fit(self, X: Tensor):
        with backend.no_grad():
            if X.ndim > 1:
                if X.shape[1] != 1:
                    raise ValueError(
                        "OneHotEncoder currently supports only a single column."
                    )
                X = X.reshape(-1)

            X_arr = X.data
            cats = xp.unique(X_arr)
            self.categories_ = cats
            self.n_categories_ = int(cats.shape[0])

        return self

    def transform(self, X: Tensor) -> Tensor:
        with backend.no_grad():
            if self.categories_ is None or self.n_categories_ is None:
                raise RuntimeError("OneHotEncoder not fitted.")

            if X.ndim > 1:
                if X.shape[1] != 1:
                    raise ValueError(
                        "OneHotEncoder currently supports only a single column."
                    )
                X = X.reshape(-1)

            X_arr = X.data
            n_samples = X_arr.shape[0]

            # map input values to category indices
            idx = xp.searchsorted(self.categories_, X_arr)
            # verify all values are known
            if (idx < 0).any() or (idx >= self.n_categories_).any():
                # searchsorted won't produce negatives but values not in cats
                # may still map to positions where categories_ != X_arr.
                mask_unknown = self.categories_[idx] != X_arr
                if xp.any(mask_unknown):
                    raise ValueError("Found unknown category in OneHotEncoder.transform.")

            # build one-hot matrix
            out = xp.zeros((n_samples, self.n_categories_), dtype=self.dtype)
            rows = xp.arange(n_samples, dtype="int64")
            out[rows, idx.astype("int64")] = 1

            return Tensor(out, dtype=self.dtype)