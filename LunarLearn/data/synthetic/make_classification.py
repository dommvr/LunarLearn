import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _allocate_counts, _normalize_weights

xp = backend.xp
DTYPE = backend.DTYPE


def make_classification(
    n_samples=100,
    n_features=20,
    n_informative=2,
    n_redundant=2,
    n_repeated=0,
    class_weights=None,
    n_classes=2,
    n_clusters_per_class=2,
    class_sep=1.0,
    flip_y=0.01,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Cluster-based synthetic classification data, inspired by sklearn.make_classification.

    Returns:
        X: (n_samples, n_features) float array
        y: (n_samples,) int array (0..n_classes-1)
    """
    if dtype is None:
        dtype = DTYPE

    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("n_informative + n_redundant + n_repeated must be <= n_features")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if n_clusters_per_class < 1:
        raise ValueError("n_clusters_per_class must be >= 1")

    rng = _get_rng(random_state)

    # class weights default: uniform
    weights = _normalize_weights(class_weights, n_classes)
    if weights is None:
        weights = [1.0 / n_classes] * n_classes
    class_counts = _allocate_counts(n_samples, weights)

    # Build informative features from Gaussian clusters
    X_inf_list = []
    y_list = []

    for c in range(n_classes):
        n_c = class_counts[c]
        if n_c == 0:
            continue

        # allocate across clusters
        cluster_counts = [n_c // n_clusters_per_class] * n_clusters_per_class
        for i in range(n_c % n_clusters_per_class):
            cluster_counts[i] += 1

        # Each class has its own "center", clusters are offsets around it
        class_center = rng.normal(loc=0.0, scale=class_sep, size=(n_informative,)).astype(dtype)

        for k, nk in enumerate(cluster_counts):
            if nk == 0:
                continue
            cluster_offset = rng.normal(loc=0.0, scale=0.5 * class_sep, size=(n_informative,)).astype(dtype)
            mean = class_center + cluster_offset
            Xk = rng.normal(loc=0.0, scale=1.0, size=(nk, n_informative)).astype(dtype)
            Xk = Xk + mean  # broadcast
            X_inf_list.append(Xk)
            y_list.append(xp.full((nk,), c, dtype=xp.int64))

    X_inf = xp.concatenate(X_inf_list, axis=0) if X_inf_list else xp.empty((0, n_informative), dtype=dtype)
    y = xp.concatenate(y_list, axis=0) if y_list else xp.empty((0,), dtype=xp.int64)

    # Redundant features: linear combos of informative
    X_parts = [X_inf]
    if n_redundant > 0:
        A = rng.normal(loc=0.0, scale=1.0, size=(n_informative, n_redundant)).astype(dtype)
        X_red = X_inf @ A
        X_parts.append(X_red)

    # Repeated features: duplicates of existing features
    if n_repeated > 0:
        base = xp.concatenate(X_parts, axis=1)
        if base.shape[1] == 0:
            raise ValueError("Cannot create repeated features with zero base features")
        idx = rng.randint(0, base.shape[1], size=(n_repeated,))
        X_rep = base[:, idx]
        X_parts.append(X_rep)

    X = xp.concatenate(X_parts, axis=1) if len(X_parts) > 1 else X_parts[0]

    # Fill remaining noise features
    n_current = X.shape[1]
    n_noise = n_features - n_current
    if n_noise > 0:
        X_noise = rng.normal(loc=0.0, scale=1.0, size=(X.shape[0], n_noise)).astype(dtype)
        X = xp.concatenate([X, X_noise], axis=1)

    # Shuffle samples and features if requested
    if shuffle:
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

        feat_perm = rng.permutation(n_features)
        X = X[:, feat_perm]

    # Flip labels
    if flip_y and flip_y > 0:
        n_flip = int(round(float(flip_y) * X.shape[0]))
        if n_flip > 0:
            flip_idx = rng.choice(X.shape[0], size=(n_flip,), replace=False)
            # assign to random incorrect class
            new_y = rng.randint(0, n_classes, size=(n_flip,), dtype=xp.int64)
            # ensure different
            same = new_y == y[flip_idx]
            # re-roll where same (one extra try is enough for typical n_classes)
            if xp.any(same):
                new_y[same] = (new_y[same] + 1) % n_classes
            y[flip_idx] = new_y

    return X.astype(dtype), y