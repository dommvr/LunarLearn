import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor
from typing import Tuple, List

xp = backend.xp

def true_positive(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class TP
        per_class_tp = xp.sum((preds_bin == 1) & (targets == 1), axis=0)
        # total TP
        total_tp = int(xp.sum(per_class_tp))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_tp = xp.zeros(C, dtype=int)
        # TP per class: prediction must match and equal the class
        matches = (preds == targets)
        for c in range(C):
            per_class_tp[c] = xp.sum(matches & (targets == c))
        
        total_tp = int(xp.sum(per_class_tp))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_tp, per_class_tp


def true_negative(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class TN
        per_class_tn = xp.sum((preds_bin == 0) & (targets == 0), axis=0)
        # total TN
        total_tn = int(xp.sum(per_class_tn))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_tn = xp.zeros(C, dtype=int)
        # TN per class: model did NOT predict class c AND true label is NOT class c
        for c in range(C):
            per_class_tn[c] = xp.sum((preds != c) & (targets != c))
        
        total_tn = int(xp.sum(per_class_tn))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_tn, per_class_tn


def false_positive(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class FP
        per_class_fp = xp.sum((preds_bin == 1) & (targets == 0), axis=0)
        # total FP
        total_fp = int(xp.sum(per_class_fp))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_fp = xp.zeros(C, dtype=int)
        # FP per class: model did predict class c AND true label is NOT class c
        for c in range(C):
            per_class_fp[c] = xp.sum((preds == c) & (targets != c))
        
        total_fp = int(xp.sum(per_class_fp))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_fp, per_class_fp


def false_negative(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class FN
        per_class_fn = xp.sum((preds_bin == 0) & (targets == 1), axis=0)
        # total FN
        total_fn = int(xp.sum(per_class_fn))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_fn = xp.zeros(C, dtype=int)
        # FN per class: model did NOT predict class c AND true label is class c
        for c in range(C):
            per_class_fn[c] = xp.sum((preds != c) & (targets == c))
        
        total_fn = int(xp.sum(per_class_fn))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_fn, per_class_fn


def _binary_auroc(scores, labels, eps=1e-12):
    # Sort by predicted score
    order = xp.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    # Count positives and negatives
    P = xp.sum(labels == 1)
    N = xp.sum(labels == 0)

    if P == 0 or N == 0:
        return 0.5  # undefined, neutral score

    # Compute rank-sum AUROC (Mann–Whitney)
    # AUROC = (sum of ranks of positive - P*(P+1)/2) / (P*N)
    ranks = xp.arange(1, len(scores) + 1)
    pos_ranks = xp.sum(ranks[labels == 1])

    auc = (pos_ranks - P * (P + 1) / 2) / (P * N + eps)
    return float(auc)


def _binary_auprc(scores, labels, eps=1e-12):
    """
    Compute AUPRC (average precision) for a binary problem.
    scores: (N,) predicted probabilities or logits
    labels: (N,) ground truth 0/1
    """

    # sort by descending score
    order = xp.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    # cumulative true positives and false positives
    tp = xp.cumsum(labels == 1)
    fp = xp.cumsum(labels == 0)

    P = xp.sum(labels == 1)
    if P == 0:
        return 0.0

    precision = tp / (tp + fp + eps)
    recall = tp / (P + eps)

    # trapezoidal integration under PR curve
    return float(xp.trapz(precision, recall))


def _count_ngrams(tokens: Tensor, n: int):
    counts = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(int(x) for x in tokens[i:i+n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _modified_precision(preds: Tensor, targets: Tensor, n: int):
    preds_ngrams = _count_ngrams(preds, n)
    tgt_ngrams = _count_ngrams(targets, n)

    clipped = 0
    total = 0

    for ng, count in preds_ngrams.items():
        total += count
        clipped += min(count, tgt_ngrams.get(ng, 0))

    if total == 0:
        return 0.0

    return clipped / total


def _lcs_length(a: Tensor, b: Tensor):
    """
    Compute length of LCS between token lists a and b.
    Classic DP O(n*m). Fine for sentence-level use.
    """
    n = len(a)
    m = len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[n][m]


def _activation_stats(acts: Tensor, eps: float = 1e-6):
    """
    Compute mean and covariance of activations.

    acts: Tensor of shape (N, D)
    Returns:
        mu: xp.ndarray, shape (D,)
        cov: xp.ndarray, shape (D, D)
    """
    x = acts.data.astype(xp.float64)
    if x.ndim != 2:
        raise ValueError("FID expects activations of shape (N, D).")
    N, D = x.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")

    mu = xp.mean(x, axis=0)

    xc = x - mu
    # unbiased covariance: (X^T X) / (N - 1)
    cov = (xc.T @ xc) / (N - 1 + eps)
    return mu, cov


def _matrix_sqrt(mat: xp.ndarray, eps: float = 1e-6) -> xp.ndarray:
    """
    Symmetric PSD matrix square root using eigen-decomposition.
    mat: (D, D), assumed symmetric.
    """
    # ensure symmetry (numerical)
    mat = 0.5 * (mat + mat.T)

    # eigen decomposition
    w, v = xp.linalg.eigh(mat)  # for symmetric
    w = xp.clip(w, a_min=0.0, a_max=None)
    sqrt_w = xp.sqrt(w + eps)

    return (v * sqrt_w[None, :]) @ v.T


def _gaussian_kernel(size: int, sigma: float):
    ax = xp.arange(size) - size // 2
    kernel = xp.exp(-(ax**2) / (2 * sigma**2))
    kernel = kernel / xp.sum(kernel)
    return kernel


def _gaussian_filter(img, size: int, sigma: float):
    # separable filter: conv along H then W
    kernel = _gaussian_kernel(size, sigma)
    kernel = kernel.reshape(1, 1, 1, size)  # horizontal
    img = xp.apply_along_axis(lambda m: xp.convolve(m, kernel[0,0,0], mode="same"), 3, img)
    kernel = kernel.transpose(0, 1, 3, 2)      # vertical
    img = xp.apply_along_axis(lambda m: xp.convolve(m, kernel[0,0,:,0], mode="same"), 2, img)
    return img


def _inception_score_split(
    logits: Tensor,
    num_splits: int = 10,
    eps: float = 1e-12
) -> Tuple[float, float, xp.ndarray]:
    """
    Split Inception Score.

    Args:
        logits: Tensor of shape (N, C)
        num_splits: number of splits (default 10)
        eps: numeric stabilizer

    Returns:
        mean_IS: float
        std_IS: float
        split_IS: xp.ndarray of shape (num_splits,)
    """
    x = logits.data
    N = x.shape[0]

    if N < num_splits:
        raise ValueError("Number of samples must be >= num_splits.")

    # Compute log-softmax ONCE
    log_probs = ops.log_softmax(logits, axis=1, epsilon=eps).data
    probs = xp.exp(log_probs)

    # Split into chunks
    split_size = N // num_splits
    split_scores = []

    for k in range(num_splits):
        start = k * split_size
        end = (k + 1) * split_size

        p_yx = probs[start:end]          # (split_size, C)
        p_y = xp.mean(p_yx, axis=0)      # (C,)
        log_p_y = xp.log(p_y + eps)

        # KL divergence for the split
        kl = xp.sum(p_yx * (log_probs[start:end] - log_p_y[None, :]), axis=1)
        kl_mean = xp.mean(kl)

        split_scores.append(float(xp.exp(kl_mean)))

    split_scores = xp.array(split_scores, dtype=xp.float32)

    mean_IS = float(xp.mean(split_scores))
    std_IS = float(xp.std(split_scores))

    return mean_IS, std_IS, split_scores


def _box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-12):
    # unwrap Tensors
    if hasattr(boxes1, "data"):
        boxes1 = boxes1.data
    if hasattr(boxes2, "data"):
        boxes2 = boxes2.data

    if boxes1.size == 0 or boxes2.size == 0:
        return xp.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=xp.float32)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # broadcast to (N, M)
    x1 = xp.maximum(x11[:, None], x21[None, :])
    y1 = xp.maximum(y11[:, None], y21[None, :])
    x2 = xp.minimum(x12[:, None], x22[None, :])
    y2 = xp.minimum(y12[:, None], y22[None, :])

    inter_w = xp.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = xp.clip(y2 - y1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)


def _mean_average_precision(
    pred_boxes: List[Tensor],
    pred_scores: List[Tensor],
    pred_labels: List[Tensor],
    gt_boxes: List[Tensor],
    gt_labels: List[Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
    eps: float = 1e-12,
) -> Tuple[float, xp.ndarray]:
    """
    VOC-style mAP@IoU threshold.

    Args:
        pred_boxes:  list of (Pi, 4) Tensors - predicted boxes per image
        pred_scores: list of (Pi,) Tensors - scores per predicted box
        pred_labels: list of (Pi,) Tensors - predicted class ids
        gt_boxes:    list of (Gi, 4) Tensors - ground truth boxes per image
        gt_labels:   list of (Gi,) Tensors - GT class ids
        num_classes: number of classes (int)
        iou_threshold: IoU threshold to consider a prediction TP
        eps: small constant

    Returns:
        mAP: float - mean of per-class AP over classes with at least one GT
        per_class_ap: xp.ndarray of shape (num_classes,)
    """

    # Count GT boxes per class
    gt_per_class = xp.zeros(num_classes, dtype=xp.int64)
    for g_lbl in gt_labels:
        g = g_lbl.data
        for c in range(num_classes):
            gt_per_class[c] += xp.sum(g == c)

    per_class_ap = xp.zeros(num_classes, dtype=xp.float32)

    # Process each class independently (one-vs-rest)
    for c in range(num_classes):
        n_gt = int(gt_per_class[c])
        if n_gt == 0:
            # no GT for this class → ignore in mAP
            per_class_ap[c] = 0.0
            continue

        # Collect predictions of class c across all images
        class_preds = []  # list of (score, image_idx, box_tensor)
        for img_idx, (p_boxes, p_scores, p_labels) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels)
        ):
            lb = p_labels.data
            mask = (lb == c)
            if xp.any(mask):
                b = p_boxes.data[mask]
                s = p_scores.data[mask]
                for j in range(b.shape[0]):
                    class_preds.append((float(s[j]), img_idx, b[j]))

        if len(class_preds) == 0:
            per_class_ap[c] = 0.0
            continue

        # Sort predictions by descending score
        class_preds.sort(key=lambda x: x[0], reverse=True)

        # Build map from image_idx -> GT boxes of class c
        gt_boxes_c = {}
        gt_used_c = {}

        for img_idx, (g_boxes, g_labels) in enumerate(zip(gt_boxes, gt_labels)):
            g_lbl = g_labels.data
            mask = (g_lbl == c)
            if xp.any(mask):
                boxes_c = g_boxes.data[mask]
                gt_boxes_c[img_idx] = boxes_c
                gt_used_c[img_idx] = xp.zeros(boxes_c.shape[0], dtype=bool)

        # Evaluate predictions as TP / FP
        tp = xp.zeros(len(class_preds), dtype=xp.float32)
        fp = xp.zeros(len(class_preds), dtype=xp.float32)

        for i, (_, img_idx, box_pred) in enumerate(class_preds):
            if img_idx not in gt_boxes_c:
                # no GT of this class in this image → FP
                fp[i] = 1.0
                continue

            g_boxes = gt_boxes_c[img_idx]
            used = gt_used_c[img_idx]

            # IoU between this pred and all GT boxes of class c in this image
            ious = box_iou(box_pred[None, :], g_boxes)[0]  # shape (G,)

            best_iou_idx = int(xp.argmax(ious))
            best_iou = float(ious[best_iou_idx])

            if best_iou >= iou_threshold and not bool(used[best_iou_idx]):
                tp[i] = 1.0
                used[best_iou_idx] = True
            else:
                fp[i] = 1.0

        # cumulative TP / FP
        tp_cum = xp.cumsum(tp)
        fp_cum = xp.cumsum(fp)

        recalls = tp_cum / (n_gt + eps)
        precisions = tp_cum / (tp_cum + fp_cum + eps)

        # AP: integrate precision–recall curve (VOC2010+ style)
        # add sentinel points
        mrec = xp.concatenate([xp.array([0.0]), recalls, xp.array([1.0])])
        mpre = xp.concatenate([xp.array([0.0]), precisions, xp.array([0.0])])

        # precision envelope
        for i in range(mpre.shape[0] - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # sum over recall changes
        idx = xp.where(mrec[1:] != mrec[:-1])[0]
        ap = xp.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        per_class_ap[c] = float(ap)

    # mAP over classes that actually appear in GT
    valid = gt_per_class > 0
    if xp.any(valid):
        mAP = float(xp.sum(per_class_ap[valid]) / (xp.sum(valid) + eps))
    else:
        mAP = 0.0

    return mAP, per_class_ap