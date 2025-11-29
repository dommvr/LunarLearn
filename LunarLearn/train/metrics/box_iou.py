import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _box_iou
from LunarLearn.core import Tensor

xp = backend.xp


class BoxIoU(BaseMetric):
    def __init__(self, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.eps = eps

    def compute(self, boxes1: Tensor, boxes2: Tensor):
        return _box_iou(boxes1, boxes2, self.eps)