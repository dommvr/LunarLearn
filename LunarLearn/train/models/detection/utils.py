import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE


def yolo_decode(pred: Tensor,
                anchors,
                num_classes: int,
                stride: int,
                img_size) -> tuple[Tensor, Tensor, Tensor]:
    """
    Decode YOLO head output for one scale.

    pred: (B, A*(5+num_classes), H, W)
    anchors: (A, 2) w,h in pixels (input image scale)
    stride: downsample factor (e.g. 8, 16, 32)
    img_size: (img_h, img_w)
    """
    B, C, H, W = pred.shape
    img_h, img_w = img_size
    A = len(anchors)

    anchors = xp.array(anchors, dtype=DTYPE)
    pred = pred.reshape(B, A, 5 + num_classes, H, W)

    tx = pred[:, :, 0, :, :]
    ty = pred[:, :, 1, :, :]
    tw = pred[:, :, 2, :, :]
    th = pred[:, :, 3, :, :]
    to = pred[:, :, 4, :, :]          # objectness logit
    tcls = pred[:, :, 5:, :, :]       # (B, A, C, H, W)

    grid_y, grid_x = xp.meshgrid(xp.arange(H), xp.arange(W), indexing="ij")
    grid_x = grid_x.reshape(1, 1, H, W).astype(DTYPE)
    grid_y = grid_y.reshape(1, 1, H, W).astype(DTYPE)

    bx = (ops.sigmoid(tx) + grid_x) * stride
    by = (ops.sigmoid(ty) + grid_y) * stride

    anchor_w = anchors[:, 0].reshape(1, A, 1, 1)
    anchor_h = anchors[:, 1].reshape(1, A, 1, 1)

    bw = ops.exp(tw) * anchor_w
    bh = ops.exp(th) * anchor_h

    x1 = bx - bw / 2.0
    y1 = by - bh / 2.0
    x2 = bx + bw / 2.0
    y2 = by + bh / 2.0

    x1 = ops.clip(x1, 0.0, img_w)
    x2 = ops.clip(x2, 0.0, img_w)
    y1 = ops.clip(y1, 0.0, img_h)
    y2 = ops.clip(y2, 0.0, img_h)

    boxes = ops.stack([x1, y1, x2, y2], axis=-1)  # (B, A, H, W, 4)
    boxes = boxes.reshape(B, A * H * W, 4)

    obj_scores = ops.sigmoid(to).reshape(B, A * H * W)
    cls_scores = ops.sigmoid(tcls)
    cls_scores = cls_scores.transpose(0, 1, 3, 4, 2)          # (B, A, H, W, C)
    cls_scores = cls_scores.reshape(B, A * H * W, num_classes)

    return boxes, obj_scores, cls_scores


def wh_iou(box_wh, anchors_wh, eps=1e-7):
    """
    IoU between a single box (w,h) and multiple anchors (A,2), ignoring center.
    box_wh: (2,)  [w, h]
    anchors_wh: (A,2)
    """
    w, h = box_wh[0], box_wh[1]

    inter_w = xp.minimum(w, anchors_wh[:, 0])
    inter_h = xp.minimum(h, anchors_wh[:, 1])
    inter_area = xp.maximum(inter_w, 0.0) * xp.maximum(inter_h, 0.0)

    area_box = w * h
    area_anchors = anchors_wh[:, 0] * anchors_wh[:, 1]
    union = area_box + area_anchors - inter_area

    iou = inter_area / (union + eps)
    return iou


def build_targets_per_scale(gt_batch,
                            anchors_per_scale,
                            strides,
                            num_classes,
                            img_size):
    """
    Build YOLO-style targets for multiple scales.

    Args:
        gt_batch: list of length B. TENSOR!
            Each element is xp.ndarray of shape (N_i, 5):
            [class_id, cx, cy, w, h] in *pixels*.
        anchors_per_scale: list of length S, each (A_s, 2) in pixels.
        strides: list of length S (e.g. [8,16,32]).
        num_classes: int.
        img_size: (img_h, img_w).

    Returns:
        targets_per_scale: list of length S.
            targets_per_scale[s].shape = (B, A_s, H_s, W_s, 5 + num_classes)
    """
    img_h, img_w = img_size
    B = len(gt_batch)
    S = len(anchors_per_scale)

    # Precompute feature map sizes per scale
    Hs = [img_h // stride for stride in strides]
    Ws = [img_w // stride for stride in strides]

    # Flatten all anchors to pick best anchor over all scales
    all_anchors = []
    anchor_scale_index = []
    anchor_index_in_scale = []
    for s, anchors_s in enumerate(anchors_per_scale):
        A_s = anchors_s.shape[0]
        all_anchors.append(anchors_s)
        for a in range(A_s):
            anchor_scale_index.append(s)
            anchor_index_in_scale.append(a)
    all_anchors = xp.vstack(all_anchors)  # (A_total, 2)
    anchor_scale_index = xp.array(anchor_scale_index, dtype=xp.int64)         # (A_total,)
    anchor_index_in_scale = xp.array(anchor_index_in_scale, dtype=xp.int64)   # (A_total,)

    # Allocate target tensors for each scale
    targets_per_scale = []
    for s in range(S):
        A_s = anchors_per_scale[s].shape[0]
        H_s = Hs[s]
        W_s = Ws[s]
        # shape: (B, A_s, H_s, W_s, 5 + C)
        t = xp.zeros((B, A_s, H_s, W_s, 5 + num_classes), dtype=DTYPE)
        targets_per_scale.append(t)

    # Fill targets
    for b, gt in enumerate(gt_batch):
        if gt is None or gt.size == 0:
            continue

        # gt: (N, 5) -> [cls, cx, cy, w, h]
        for n in range(gt.shape[0]):
            cls = int(gt[n, 0])
            cx  = float(gt[n, 1])
            cy  = float(gt[n, 2])
            w   = float(gt[n, 3])
            h   = float(gt[n, 4])

            # skip weird boxes
            if w <= 0 or h <= 0:
                continue

            # clamp centers inside image
            if cx < 0 or cy < 0 or cx >= img_w or cy >= img_h:
                continue

            # pick best anchor across all scales
            box_wh = xp.array([w, h], dtype=DTYPE)
            ious = wh_iou(box_wh, all_anchors)        # (A_total,)
            best_idx = int(xp.argmax(ious))

            s = int(anchor_scale_index[best_idx])     # which scale
            a = int(anchor_index_in_scale[best_idx])  # which anchor in that scale

            stride_s = strides[s]
            H_s = Hs[s]
            W_s = Ws[s]

            # Map center to grid cell indices
            gx = cx / stride_s
            gy = cy / stride_s
            j = int(gx)   # col (x)
            i = int(gy)   # row (y)

            if i < 0 or j < 0 or i >= H_s or j >= W_s:
                continue  # safety

            # Compute corners in pixels
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0

            # clamp to image just in case
            x1 = max(0.0, min(x1, img_w))
            x2 = max(0.0, min(x2, img_w))
            y1 = max(0.0, min(y1, img_h))
            y2 = max(0.0, min(y2, img_h))

            t_s = targets_per_scale[s]

            # objectness
            t_s[b, a, i, j, 0] = 1.0

            # box
            t_s[b, a, i, j, 1] = x1
            t_s[b, a, i, j, 2] = y1
            t_s[b, a, i, j, 3] = x2
            t_s[b, a, i, j, 4] = y2

            # class one-hot
            if 0 <= cls < num_classes:
                t_s[b, a, i, j, 5 + cls] = 1.0

    # Wrap into Tensor if you want
    targets_per_scale = [Tensor(t, requires_grad=False, dtype=DTYPE)
                         for t in targets_per_scale]

    return targets_per_scale
