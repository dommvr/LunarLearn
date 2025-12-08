
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
    return classes, y_encoded.astype("int64")


class _TreeNode:
    __slots__ = ("is_leaf", "feature", "threshold", "left", "right", "value")

    def __init__(self,
                 is_leaf: bool = True,
                 feature: int | None = None,
                 threshold: float | None = None,
                 left: "_TreeNode | None" = None,
                 right: "_TreeNode | None" = None,
                 value=None):
        self.is_leaf = is_leaf
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # classifier: probs (C,), regressor: scalar