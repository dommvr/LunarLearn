import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor

xp = backend.xp


def _encode_labels(y: Tensor):
    """
    Encode arbitrary labels into 0..n_classes-1 and return (classes, encoded).

    Parameters
    ----------
    y : Tensor of shape (n_samples,)

    Returns
    -------
    classes : xp.ndarray shape (n_classes,)
    y_encoded : xp.ndarray shape (n_samples,), int64
    """
    y_arr = y.data
    classes = xp.unique(y_arr)
    y_encoded = xp.searchsorted(classes, y_arr)
    return classes, y_encoded.astype("int64")