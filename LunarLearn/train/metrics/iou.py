import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor

xp = backend.xp


class IoU(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.threshold = threshold
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor):
        """
        Intersection over Union (IoU) for segmentation.

        Supports:
        - preds: (N, C, H, W), targets: (N, H, W)       → multi-class logits + class indices
        - preds: (N, C, H, W), targets: (N, C, H, W)    → multi-label / one-hot masks
        - preds: (N, H, W),    targets: (N, H, W)       → binary segmentation

        Returns:
            micro_iou: float
            macro_iou: float
            weighted_iou: float
            per_class_iou: xp.ndarray of shape (C,)
        """
        p = preds.data
        t = targets.data

        # ---------- CASE 1: logits (N, C, H, W) + class indices (N, H, W) ----------
        if p.ndim == 4 and t.ndim == 3:
            N, C, H, W = p.shape

            # argmax over class dimension -> predicted labels
            pred_labels = xp.argmax(p, axis=1)   # (N, H, W)
            true_labels = t

            # flatten
            pred_flat = pred_labels.reshape(-1)
            true_flat = true_labels.reshape(-1)

            # number of classes
            C = int(max(pred_flat.max(), true_flat.max()) + 1)

            inter = xp.zeros(C, dtype=xp.int64)
            union = xp.zeros(C, dtype=xp.int64)
            support = xp.zeros(C, dtype=xp.int64)   # number of GT pixels per class

            for c in range(C):
                pred_c = (pred_flat == c)
                true_c = (true_flat == c)

                inter[c] = xp.sum(pred_c & true_c)
                union[c] = xp.sum(pred_c | true_c)
                support[c] = xp.sum(true_c)

            per_class_iou = inter / (union + self.eps)
            micro_iou = float(xp.sum(inter) / (xp.sum(union) + self.eps))
            macro_iou = float(xp.mean(per_class_iou))

            total_support = xp.sum(support) + self.eps
            weighted_iou = float(xp.sum(per_class_iou * support) / total_support)

            return micro_iou, macro_iou, weighted_iou, per_class_iou

        # ---------- CASE 2: masks with same shape ---------- 
        # (N, C, H, W) multi-label or (N, H, W) binary
        if p.shape != t.shape:
            raise ValueError("For mask IoU, preds and targets must have the same shape or (N,C,H,W)/(N,H,W).")

        # If not integer, threshold predictions
        if p.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (p > self.threshold).astype(xp.int32)
        else:
            preds_bin = p

        # Targets: treat any non-zero as 1
        if t.dtype not in [xp.int32, xp.int64, xp.uint8]:
            targets_bin = (t > 0.5).astype(xp.int32)
        else:
            targets_bin = (t != 0).astype(xp.int32)

        if preds_bin.ndim == 3:
            # (N, H, W) → single "class"
            preds_flat = preds_bin.reshape(1, -1)
            targets_flat = targets_bin.reshape(1, -1)
            C = 1
        elif preds_bin.ndim == 4:
            # (N, C, H, W) → multi-label / multi-class one-hot
            N, C, H, W = preds_bin.shape
            preds_flat = preds_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
            targets_flat = targets_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
        else:
            raise ValueError("Unsupported mask shape for IoU: expected (N,H,W) or (N,C,H,W).")

        # intersection & union per class
        intersection = xp.sum((preds_flat == 1) & (targets_flat == 1), axis=1)
        union = xp.sum((preds_flat == 1) | (targets_flat == 1), axis=1)
        support = xp.sum(targets_flat == 1, axis=1)

        per_class_iou = intersection / (union + self.eps)
        micro_iou = float(xp.sum(intersection) / (xp.sum(union) + self.eps))
        macro_iou = float(xp.mean(per_class_iou))

        total_support = xp.sum(support) + self.eps
        weighted_iou = float(xp.sum(per_class_iou * support) / total_support)

        return micro_iou, macro_iou, weighted_iou, per_class_iou