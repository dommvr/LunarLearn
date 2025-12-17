import LunarLearn.core.backend.backend as backend
from LunarLearn.data.augmentation.utils import (_rand,
                                                _randint,
                                                _as_xp,
                                                _safe_std,
                                                _ensure_2d,
                                                _knn_indices,
                                                _unique_counts_1d,
                                                _sample_indices_weighted)

xp = backend.xp
DTYPE = backend.DTYPE


# ----------------------------
# 1) Numeric feature perturbations
# ----------------------------

class GaussianJitter:
    """
    Add Gaussian noise per feature: x += N(0, (sigma*std)^2)
    - If feature_std not provided, uses std from this sample (not great) unless you pass global stats.
    Best practice: pass feature_std computed on training set.
    """
    def __init__(self, sigma=0.05, feature_std=None, p=0.5, eps=1e-8, dtype=None):
        self.sigma = float(sigma)
        self.p = float(p)
        self.eps = float(eps)
        self.dtype = dtype
        self.feature_std = None if feature_std is None else xp.asarray(feature_std)

    def __call__(self, x):
        if _rand() > self.p:
            return x
        X = _as_xp(x, dtype=self.dtype)
        if X.ndim != 1:
            raise ValueError("GaussianJitter expects a single row x with shape (F,)")

        std = self.feature_std
        if std is None:
            # fallback: per-row std (meh but ok for playground)
            std = _safe_std(X[None, :], axis=0, eps=self.eps)

        noise = xp.random.normal(0.0, 1.0, size=X.shape).astype(X.dtype)
        return X + noise * (self.sigma * std)


class FeatureScalingJitter:
    """
    Multiply features by noise ~ N(1, sigma^2) per feature.
    """
    def __init__(self, sigma=0.03, p=0.5, dtype=None):
        self.sigma = float(sigma)
        self.p = float(p)
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        X = _as_xp(x, dtype=self.dtype)
        if X.ndim != 1:
            raise ValueError("FeatureScalingJitter expects x shape (F,)")
        m = xp.random.normal(1.0, self.sigma, size=X.shape).astype(X.dtype)
        return X * m


class QuantizationNoise:
    """
    Simulate sensor rounding: x -> round(x / step) * step
    - step can be scalar or per-feature array
    - if step is None and feature_std provided, step = q * std
    """
    def __init__(self, step=None, q=0.02, feature_std=None, p=0.3, dtype=None, eps=1e-8):
        self.step = None if step is None else xp.asarray(step)
        self.q = float(q)
        self.feature_std = None if feature_std is None else xp.asarray(feature_std)
        self.p = float(p)
        self.dtype = dtype
        self.eps = float(eps)

    def __call__(self, x):
        if _rand() > self.p:
            return x
        X = _as_xp(x, dtype=self.dtype)
        if X.ndim != 1:
            raise ValueError("QuantizationNoise expects x shape (F,)")

        step = self.step
        if step is None:
            if self.feature_std is None:
                # fallback: global-ish from row
                s = _safe_std(X[None, :], axis=0, eps=self.eps)
            else:
                s = xp.maximum(self.feature_std, self.eps)
            step = self.q * s

        step = xp.maximum(step.astype(X.dtype), xp.asarray(self.eps, dtype=X.dtype))
        return xp.round(X / step) * step


# ----------------------------
# 2) Missingness / corruption
# ----------------------------

class FeatureDropout:
    """
    Set random features to missing_value (or 0) with probability p_feat.
    """
    def __init__(self, p_feat=0.05, missing_value=0.0, p=0.5, dtype=None):
        self.p_feat = float(p_feat)
        self.missing_value = missing_value
        self.p = float(p)
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        X = _as_xp(x, dtype=self.dtype).copy()
        if X.ndim != 1:
            raise ValueError("FeatureDropout expects x shape (F,)")

        mask = xp.random.rand(X.shape[0]) < self.p_feat
        mv = self.missing_value
        # allow NaN missingness
        mv = xp.asarray(mv, dtype=X.dtype)
        X[mask] = mv
        return X


class RandomImputationNoise:
    """
    If you use an imputer (mean/median), add noise around the imputed value.
    Only applies where x == missing_value (or is NaN).
    """
    def __init__(self, impute_values, noise_sigma=0.02, missing_value=0.0, p=0.5, dtype=None):
        self.impute = xp.asarray(impute_values)  # (F,)
        self.sigma = float(noise_sigma)
        self.missing_value = missing_value
        self.p = float(p)
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        X = _as_xp(x, dtype=self.dtype).copy()
        if X.ndim != 1:
            raise ValueError("RandomImputationNoise expects x shape (F,)")

        imp = self.impute.astype(X.dtype)
        if xp.isnan(xp.asarray(self.missing_value)).item():
            miss = xp.isnan(X)
        else:
            miss = (X == xp.asarray(self.missing_value, dtype=X.dtype))

        if xp.any(miss):
            noise = xp.random.normal(0.0, self.sigma, size=X.shape).astype(X.dtype)
            X = xp.where(miss, imp + noise, X)
        return X
    

# ----------------------------
# 3) Resampling / synthetic minority
# ----------------------------

def bootstrap_resample(X, y=None, n_samples=None):
    """
    Bootstrap rows with replacement.
    Returns (Xb, yb) or Xb if y is None.
    """
    X = _ensure_2d(X)
    N = int(X.shape[0])
    if n_samples is None:
        n_samples = N
    idx = xp.random.randint(0, N, size=(int(n_samples),))
    Xb = X[idx]
    if y is None:
        return Xb
    y = xp.asarray(y)
    return Xb, y[idx]


def smote_resample(X, y, minority_class=None, k=5, n_samples=None):
    """
    SMOTE for binary or multiclass.
    - Select a minority class (auto: smallest count).
    - Generate synthetic points along segments between minority neighbors.

    Returns (X_new, y_new) where new includes generated points.
    """
    X = _ensure_2d(X).astype(DTYPE)
    y = xp.asarray(y).astype(xp.int64)

    classes = xp.unique(y)
    counts = xp.asarray([xp.sum(y == c) for c in classes], dtype=xp.int64)

    if minority_class is None:
        minority_class = int(classes[int(xp.argmin(counts))])

    Xmin = X[y == minority_class]
    M = int(Xmin.shape[0])
    if M < 2:
        return X, y

    if n_samples is None:
        # default: balance to majority class
        maj = int(xp.max(counts))
        n_samples = max(0, maj - M)

    if n_samples <= 0:
        return X, y

    nn = _knn_indices(Xmin, k=min(int(k), M - 1))  # (M,k)

    gen = xp.empty((int(n_samples), int(X.shape[1])), dtype=X.dtype)
    for i in range(int(n_samples)):
        a = _randint(0, M)
        b = int(nn[a, _randint(0, int(nn.shape[1]))])
        lam = float(xp.random.rand())
        gen[i] = Xmin[a] + lam * (Xmin[b] - Xmin[a])

    Xn = xp.concatenate([X, gen], axis=0)
    yn = xp.concatenate([y, xp.full((int(n_samples),), int(minority_class), dtype=xp.int64)], axis=0)
    return Xn, yn


def borderline_smote_resample(X, y, minority_class=None, k=5, m=10, n_samples=None):
    """
    Borderline-SMOTE (simple version):
    - find minority points with many majority neighbors (borderline)
    - generate SMOTE samples from those points

    k: neighbors for synthesis within minority
    m: neighbors for borderline test over all data
    """
    X = _ensure_2d(X).astype(DTYPE)
    y = xp.asarray(y).astype(xp.int64)

    classes = xp.unique(y)
    counts = xp.asarray([xp.sum(y == c) for c in classes], dtype=xp.int64)

    if minority_class is None:
        minority_class = int(classes[int(xp.argmin(counts))])

    Xmin = X[y == minority_class]
    Mmin = int(Xmin.shape[0])
    if Mmin < 2:
        return X, y

    if n_samples is None:
        maj = int(xp.max(counts))
        n_samples = max(0, maj - Mmin)
    if n_samples <= 0:
        return X, y

    # brute-force neighbors over all data for each minority sample
    # (Mmin, N)
    dif = Xmin[:, None, :] - X[None, :, :]
    d2 = xp.sum(dif * dif, axis=2)
    nn_all = xp.argsort(d2, axis=1)[:, 1:int(m)+1]  # exclude self-ish

    # borderline if many neighbors are not minority
    neigh_labels = y[nn_all]  # (Mmin,m)
    maj_count = xp.sum(neigh_labels != minority_class, axis=1)
    borderline = maj_count >= (int(m) // 2)
    idx_border = xp.where(borderline)[0]
    if int(idx_border.shape[0]) == 0:
        # fallback to normal SMOTE
        return smote_resample(X, y, minority_class=minority_class, k=k, n_samples=n_samples)

    # neighbors within minority for synthesis
    nn_min = _knn_indices(Xmin, k=min(int(k), Mmin - 1))

    gen = xp.empty((int(n_samples), int(X.shape[1])), dtype=X.dtype)
    for i in range(int(n_samples)):
        a = int(idx_border[_randint(0, int(idx_border.shape[0]))])
        b = int(nn_min[a, _randint(0, int(nn_min.shape[1]))])
        lam = float(xp.random.rand())
        gen[i] = Xmin[a] + lam * (Xmin[b] - Xmin[a])

    Xn = xp.concatenate([X, gen], axis=0)
    yn = xp.concatenate([y, xp.full((int(n_samples),), int(minority_class), dtype=xp.int64)], axis=0)
    return Xn, yn


def adasyn_resample(X, y, minority_class=None, m=10, n_samples=None):
    """
    ADASYN (simple version):
    - allocate more synthetic samples to minority points surrounded by majority
    - synthesis similar to SMOTE along minority-neighbor lines

    m: neighbors over all data for difficulty
    """
    X = _ensure_2d(X).astype(DTYPE)
    y = xp.asarray(y).astype(xp.int64)

    classes = xp.unique(y)
    counts = xp.asarray([xp.sum(y == c) for c in classes], dtype=xp.int64)

    if minority_class is None:
        minority_class = int(classes[int(xp.argmin(counts))])

    Xmin = X[y == minority_class]
    Mmin = int(Xmin.shape[0])
    if Mmin < 2:
        return X, y

    if n_samples is None:
        maj = int(xp.max(counts))
        n_samples = max(0, maj - Mmin)
    if n_samples <= 0:
        return X, y

    # difficulty ratio ri for each minority point based on m-NN in full data
    dif = Xmin[:, None, :] - X[None, :, :]
    d2 = xp.sum(dif * dif, axis=2)
    nn_all = xp.argsort(d2, axis=1)[:, 1:int(m)+1]
    neigh_labels = y[nn_all]
    ri = xp.mean(neigh_labels != minority_class, axis=1).astype(DTYPE)  # (Mmin,)
    s = float(ri.sum())
    if s <= 0:
        # fallback to SMOTE
        return smote_resample(X, y, minority_class=minority_class, k=5, n_samples=n_samples)

    gi = ri / s  # proportions
    # integer allocation
    alloc = xp.floor(gi * int(n_samples)).astype(xp.int64)
    # fix remainder
    rem = int(n_samples) - int(alloc.sum())
    if rem > 0:
        # give remainder to highest ri
        order = xp.argsort(-ri)
        for j in range(rem):
            alloc[int(order[j % int(order.shape[0])])] += 1

    # minority knn for synthesis
    nn_min = _knn_indices(Xmin, k=min(5, Mmin - 1))

    gens = []
    for a in range(Mmin):
        na = int(alloc[a])
        if na <= 0:
            continue
        for _ in range(na):
            b = int(nn_min[a, _randint(0, int(nn_min.shape[1]))])
            lam = float(xp.random.rand())
            gens.append((Xmin[a] + lam * (Xmin[b] - Xmin[a]))[None, :])

    if not gens:
        return X, y

    gen = xp.concatenate(gens, axis=0)
    Xn = xp.concatenate([X, gen], axis=0)
    yn = xp.concatenate([y, xp.full((int(gen.shape[0]),), int(minority_class), dtype=xp.int64)], axis=0)
    return Xn, yn


# ----------------------------
# 4) Mixing (batch-level)
# ----------------------------

def tabular_mixup_batch(X, y, alpha=0.2):
    """
    Batch-level MixUp for tabular.
    X: (N,F)
    y: (N,) (regression) or (N,K) (soft labels) or (N,) class ids (you probably want one-hot)
    """
    if alpha <= 0:
        return X, y

    X = _ensure_2d(X).astype(DTYPE)
    y = xp.asarray(y)

    N = int(X.shape[0])
    lam = float(xp.random.beta(alpha, alpha))
    perm = xp.random.permutation(N)

    X2 = lam * X + (1.0 - lam) * X[perm]

    # y handling
    if y.ndim == 1:
        # regression or hard labels. For classification hard labels, this creates soft-ish targets (not great unless you one-hot).
        y2 = lam * y.astype(DTYPE) + (1.0 - lam) * y[perm].astype(DTYPE)
    else:
        y2 = lam * y.astype(DTYPE) + (1.0 - lam) * y[perm].astype(DTYPE)

    return X2, y2


# ----------------------------
# 5) Category-safe augmentations (encoded categories)
# ----------------------------

class CategorySwapWithinClass:
    """
    Swap categorical columns between samples of the same class.
    Assumes categories are integer-encoded in X.

    Works best as a dataset-level transform that has access to global (X,y).
    This class operates on a single row, but needs reference pools.
    """
    def __init__(self, X_ref, y_ref, cat_idx, p_row=0.3, p_cat=0.5):
        self.Xr = _ensure_2d(X_ref)
        self.yr = xp.asarray(y_ref).astype(xp.int64)
        self.cat_idx = [int(i) for i in cat_idx]
        self.p_row = float(p_row)
        self.p_cat = float(p_cat)

    def __call__(self, sample):
        # sample can be (x,y) or dict with keys
        if isinstance(sample, dict):
            x = sample["x"] if "x" in sample else sample.get("X", None)
            y = sample.get("y", sample.get("label", None))
            if x is None or y is None:
                raise KeyError("dict sample must contain x/X and y/label for CategorySwapWithinClass")
            x2, y2 = self._swap(x, y)
            out = dict(sample)
            out["x"] = x2
            out["y"] = y2
            return out

        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("CategorySwapWithinClass expects (x,y) or dict sample")
        x2, y2 = self._swap(sample[0], sample[1])
        return (x2, y2) if len(sample) == 2 else (x2, y2, *sample[2:])

    def _swap(self, x, y):
        if _rand() > self.p_row:
            return x, y
        x = xp.asarray(x).copy()
        cls = int(xp.asarray(y).astype(xp.int64))

        pool_idx = xp.where(self.yr == cls)[0]
        if int(pool_idx.shape[0]) <= 1:
            return x, y

        j = int(pool_idx[_randint(0, int(pool_idx.shape[0]))])
        donor = self.Xr[j]

        for ci in self.cat_idx:
            if _rand() < self.p_cat:
                x[ci] = donor[ci]
        return x, y


def rare_category_sample_weights(X, cat_idx, power=1.0, eps=1e-6):
    """
    Compute per-row sampling weights to upweight rare categories.
    Assumes categories are integer-encoded.

    weight(row) = mean_i (1 / freq(category_i)^power)
    """
    X = _ensure_2d(X)
    cat_idx = [int(i) for i in cat_idx]

    N = int(X.shape[0])
    w = xp.zeros((N,), dtype=DTYPE)

    for ci in cat_idx:
        col = X[:, ci].astype(xp.int64)
        u, c = _unique_counts_1d(col)
        # map category -> freq
        # compute freq per row by matching u
        freq = xp.zeros((N,), dtype=DTYPE)
        for ui in range(int(u.shape[0])):
            mask = (col == u[ui])
            freq[mask] = float(c[ui])
        w += 1.0 / xp.power(freq + eps, power)

    w = w / float(len(cat_idx)) if cat_idx else xp.ones((N,), dtype=DTYPE)
    return w


def rare_category_resample(X, y=None, cat_idx=None, n_samples=None, power=1.0):
    """
    Oversample rows with rare categories using weighted sampling.
    """
    X = _ensure_2d(X)
    N = int(X.shape[0])
    if n_samples is None:
        n_samples = N
    if cat_idx is None:
        # fallback: uniform bootstrap
        return bootstrap_resample(X, y, n_samples=n_samples) if y is not None else bootstrap_resample(X, None, n_samples=n_samples)

    w = rare_category_sample_weights(X, cat_idx=cat_idx, power=power)
    idx = _sample_indices_weighted(w, n=int(n_samples))
    Xr = X[idx]
    if y is None:
        return Xr
    y = xp.asarray(y)
    return Xr, y[idx]