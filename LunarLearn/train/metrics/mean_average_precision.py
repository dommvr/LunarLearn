import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _mean_average_precision
from LunarLearn.core import Tensor
from typing import List, Tuple

xp = backend.xp


class MeanAveragePrecision(BaseMetric):
    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        eps: float = 1e-12,
    ):
        # we’re not using BaseMetric's scalar/vector accumulation here,
        # just the API surface (__call__, reset, etc.)
        super().__init__(expect_vector=False)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.eps = eps
        # buffers
        self._pred_boxes: List[Tensor] = []
        self._pred_scores: List[Tensor] = []
        self._pred_labels: List[Tensor] = []
        self._gt_boxes: List[Tensor] = []
        self._gt_labels: List[Tensor] = []

    def reset(self):
        super().reset()
        self._pred_boxes = []
        self._pred_scores = []
        self._pred_labels = []
        self._gt_boxes = []
        self._gt_labels = []

    def update(
        self,
        pred_boxes: List[Tensor],
        pred_scores: List[Tensor],
        pred_labels: List[Tensor],
        gt_boxes: List[Tensor],
        gt_labels: List[Tensor],
        **kwargs,
    ):
        """
        Accumulate batch detections and ground truth.

        Each argument is a list over images in the batch:
          - pred_boxes[i]: (Pi, 4)
          - pred_scores[i]: (Pi,)
          - pred_labels[i]: (Pi,)
          - gt_boxes[i]: (Gi, 4)
          - gt_labels[i]: (Gi,)
        """
        self._pred_boxes.extend(pred_boxes)
        self._pred_scores.extend(pred_scores)
        self._pred_labels.extend(pred_labels)
        self._gt_boxes.extend(gt_boxes)
        self._gt_labels.extend(gt_labels)
        self.count += 1  # just for bookkeeping

    def compute(
        self,
        pred_boxes: List[Tensor],
        pred_scores: List[Tensor],
        pred_labels: List[Tensor],
        gt_boxes: List[Tensor],
        gt_labels: List[Tensor],
        **kwargs,
    ) -> Tuple[float, xp.ndarray]:
        """
        One-shot compute: use if you already have all detections/GT.
        """
        return _mean_average_precision(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            eps=self.eps,
        )

    def value(self) -> Tuple[float, xp.ndarray]:
        """
        Compute mAP over all accumulated detections/GT.
        """
        if len(self._pred_boxes) == 0:
            # no data, return zero and empty per-class
            return 0.0, xp.zeros(self.num_classes, dtype=xp.float32)

        mAP, per_class_ap = _mean_average_precision(
            pred_boxes=self._pred_boxes,
            pred_scores=self._pred_scores,
            pred_labels=self._pred_labels,
            gt_boxes=self._gt_boxes,
            gt_labels=self._gt_labels,
            num_classes=self.num_classes,
            iou_threshold=self.iou_threshold,
            eps=self.eps,
        )
        return mAP, per_class_ap
