import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _flip_labels, _normalize_weights, _allocate_counts, _sample_values

xp = backend.xp
DTYPE = backend.DTYPE


def make_sparse_classification(
    n_samples=3000,
    n_features=20000,
    n_informative=50,
    n_classes=2,
    density=0.001,              # fraction of nonzeros per row, ~k = density*n_features
    class_weights=None,         # list length n_classes
    flip_y=0.0,
    distribution="lognormal",   # poisson/exponential/lognormal/normal
    nonnegative=True,           # keep values >= 0 (good for MultinomialNB)
    boost=4.0,                  # how strongly "class-specific" informative features are boosted
    tfidf_like=True,            # apply log1p + L2 normalize rows
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    High-dimensional sparse (mostly-zero) dense matrix generator for classification.
    Returns:
        X: (n_samples, n_features) dense with many zeros
        y: (n_samples,) int labels
        informative_idx: (n_informative,) indices used as informative features
    """
    if dtype is None:
        dtype = DTYPE
    if n_features < 1:
        raise ValueError("n_features must be >= 1")
    if n_informative < 1 or n_informative > n_features:
        raise ValueError("n_informative must be in [1, n_features]")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if density <= 0 or density > 1:
        raise ValueError("density must be in (0, 1]")

    rng = _get_rng(random_state)

    weights = _normalize_weights(class_weights, n_classes)
    if weights is None:
        weights = [1.0 / n_classes] * n_classes

    # Generate labels with desired class distribution
    class_counts = _allocate_counts(n_samples, weights)
    y_parts = [xp.full((class_counts[c],), c, dtype=xp.int64) for c in range(n_classes) if class_counts[c] > 0]
    y = xp.concatenate(y_parts, axis=0) if y_parts else xp.empty((0,), dtype=xp.int64)

    if shuffle:
        perm = rng.permutation(y.shape[0])
        y = y[perm]

    # Informative feature indices
    informative_idx = rng.choice(n_features, size=(n_informative,), replace=False).astype(xp.int64)

    # Each informative feature is "owned" by one class (simple, effective)
    preferred_class = xp.arange(n_informative, dtype=xp.int64) % n_classes

    X = xp.zeros((n_samples, n_features), dtype=dtype)

    # expected nnz per row
    mean_k = float(density) * float(n_features)

    for i in range(n_samples):
        # sample k with a bit of randomness; clamp to [1, n_features]
        k = int(rng.poisson(lam=max(mean_k, 1.0)))
        if k < 1:
            k = 1
        if k > n_features:
            k = n_features

        cols = rng.choice(n_features, size=(k,), replace=False).astype(xp.int64)
        vals = _sample_values(rng, k, distribution, dtype, nonnegative=nonnegative)

        # Boost informative features that match the sample's class
        # Find which selected cols are informative
        # (cheap way: use membership check via sort + searchsorted)
        inf_sorted = xp.sort(informative_idx)
        pos = xp.searchsorted(inf_sorted, cols)
        mask = (pos < n_informative) & (inf_sorted[pos] == cols)

        if xp.any(mask):
            # map selected informative cols back to their rank in informative_idx
            # build inverse map once (small: n_informative), use search per match
            # We can do rank lookup via searchsorted on informative_idx sorted
            # then recover rank by searching in original informative_idx.
            inf_cols = cols[mask]
            # rank in sorted informative list:
            rank_sorted = xp.searchsorted(inf_sorted, inf_cols)
            # determine which class prefers that informative feature:
            pref = preferred_class[rank_sorted]
            c = int(y[i])
            boost_mask = (pref == c)
            # apply stronger boost for matching class; mild boost for other classes
            vals_inf = vals[mask]
            vals_inf = vals_inf * xp.where(boost_mask, xp.asarray(boost, dtype=dtype), xp.asarray(1.0, dtype=dtype))
            vals[mask] = vals_inf

        X[i, cols] = vals

    if tfidf_like:
        # log1p
        X = xp.log1p(xp.maximum(X, xp.asarray(0, dtype=dtype)))
        # L2 normalize rows
        denom = xp.sqrt(xp.sum(X * X, axis=1, keepdims=True)) + xp.asarray(1e-12, dtype=dtype)
        X = X / denom

    # Label noise
    y = _flip_labels(y, n_classes=n_classes, flip_y=flip_y, rng=rng)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y, informative_idx