import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor
import math

xp = backend.xp


def _resolve_max_features(n_features: int, max_features):
    """
    Resolve max_features spec into an integer m in [1, n_features].

    max_features can be:
      - None: use all features
      - int: that many features
      - float in (0, 1]: fraction of features
      - "sqrt": sqrt(n_features)
      - "log2": log2(n_features)
    """
    if max_features is None:
        m = n_features
    elif isinstance(max_features, int):
        m = max_features
    elif isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError("max_features float must be in (0, 1].")
        m = int(max_features * n_features)
    elif isinstance(max_features, str):
        key = max_features.lower()
        if key == "sqrt":
            m = int(math.sqrt(n_features))
        elif key == "log2":
            m = int(math.log2(n_features))
        else:
            raise ValueError(f"Unsupported max_features string: {max_features}")
    else:
        raise TypeError(f"Unsupported type for max_features: {type(max_features)}")

    m = max(1, min(m, n_features))
    return m


def _encode_labels(y: Tensor):
    """
    Encode arbitrary labels into 0..n_classes-1 and return (classes, encoded).
    """
    y_arr = y.data
    classes = xp.unique(y_arr)
    y_encoded = xp.searchsorted(classes, y_arr)
    return classes, y_encoded.astype("int64")