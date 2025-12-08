import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor

xp = backend.xp


def _encode_labels(y: Tensor):
    """
    Encode arbitrary labels into 0..n_classes-1 and return (classes, encoded).
    """
    y_arr = y.data
    classes = xp.unique(y_arr)
    y_encoded = xp.searchsorted(classes, y_arr)
    return classes, y_encoded