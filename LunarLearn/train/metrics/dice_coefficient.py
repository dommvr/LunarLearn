import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor

xp = backend.xp


class DiceCoefficient(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-12):
        super().__init__(expect_vector=True)
        self.threshold = threshold
        self.eps = eps

    def compute(self, preds: Tensor, targets: Tensor):
        """
        Dice coefficient for segmentation.

        Supports:
        1) Multi-class with logits:
            preds:   (N, C, H, W)  - raw scores / logits
            targets: (N, H, W)     - integer class labels

        2) Binary / multi-label masks:
            preds:   (N, H, W)     or (N, C, H, W)
            targets: same shape as preds

        Returns:
            micro_dice: float
            macro_dice: float
            weighted_dice: float
            per_class_dice: xp.ndarray of shape (C,)
        """
        p = preds.data
        t = targets.data

        # ---------------- CASE 1: logits (N, C, H, W) + class indices (N, H, W) ----------------
        if p.ndim == 4 and t.ndim == 3:
            N, C, H, W = p.shape

            # predicted labels
            pred_labels = xp.argmax(p, axis=1)  # (N, H, W)
            true_labels = t                     # (N, H, W)

            # flatten
            pred_flat = pred_labels.reshape(-1)
            true_flat = true_labels.reshape(-1)

            C = int(max(pred_flat.max(), true_flat.max()) + 1)

            intersection = xp.zeros(C, dtype=xp.int64)
            pred_sum = xp.zeros(C, dtype=xp.int64)
            true_sum = xp.zeros(C, dtype=xp.int64)

            for c in range(C):
                pred_c = (pred_flat == c)
                true_c = (true_flat == c)

                intersection[c] = xp.sum(pred_c & true_c)
                pred_sum[c] = xp.sum(pred_c)
                true_sum[c] = xp.sum(true_c)

            per_class_dice = (2 * intersection) / (pred_sum + true_sum + self.eps)

            # micro Dice over all classes
            total_inter = xp.sum(intersection)
            total_pred = xp.sum(pred_sum)
            total_true = xp.sum(true_sum)
            micro_dice = float(2 * total_inter / (total_pred + total_true + self.eps))

            macro_dice = float(xp.mean(per_class_dice))

            # weighted by GT pixels per class
            support = true_sum
            total_support = xp.sum(support) + self.eps
            weighted_dice = float(xp.sum(per_class_dice * support) / total_support)

            return micro_dice, macro_dice, weighted_dice, per_class_dice

        # ---------------- CASE 2: masks with same shape ----------------
        if p.shape != t.shape:
            raise ValueError(
                "For mask Dice, preds and targets must have the same shape "
                "or be (N,C,H,W)/(N,H,W) for logits+labels."
            )

        # threshold predictions if not integer
        if p.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (p > self.threshold).astype(xp.int32)
        else:
            preds_bin = p

        # binarize targets: any non-zero = 1
        if t.dtype not in [xp.int32, xp.int64, xp.uint8]:
            targets_bin = (t > 0.5).astype(xp.int32)
        else:
            targets_bin = (t != 0).astype(xp.int32)

        # (N, H, W) → single class
        if preds_bin.ndim == 3:
            preds_flat = preds_bin.reshape(1, -1)
            targets_flat = targets_bin.reshape(1, -1)
            C = 1

        # (N, C, H, W) → multi-label / multi-class one-hot
        elif preds_bin.ndim == 4:
            N, C, H, W = preds_bin.shape
            preds_flat = preds_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
            targets_flat = targets_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
        else:
            raise ValueError("Unsupported mask shape for Dice: expected (N,H,W) or (N,C,H,W).")

        intersection = xp.sum((preds_flat == 1) & (targets_flat == 1), axis=1)
        pred_sum = xp.sum(preds_flat == 1, axis=1)
        true_sum = xp.sum(targets_flat == 1, axis=1)

        per_class_dice = (2 * intersection) / (pred_sum + true_sum + self.eps)

        total_inter = xp.sum(intersection)
        total_pred = xp.sum(pred_sum)
        total_true = xp.sum(true_sum)

        micro_dice = float(2 * total_inter / (total_pred + total_true + self.eps))
        macro_dice = float(xp.mean(per_class_dice))

        support = true_sum
        total_support = xp.sum(support) + self.eps
        weighted_dice = float(xp.sum(per_class_dice * support) / total_support)

        return micro_dice, macro_dice, weighted_dice, per_class_dice