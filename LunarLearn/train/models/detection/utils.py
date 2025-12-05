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
            cx = float(gt[n, 1])
            cy = float(gt[n, 2])
            w  = float(gt[n, 3])
            h = float(gt[n, 4])

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


def _generate_anchors(anchor_sizes, aspect_ratios, feat_stride, feat_h, feat_w):
    """
    Generate anchors for a feature map of size (feat_h, feat_w).

    Returns:
        anchors: (N, 4) xp.ndarray with (x1, y1, x2, y2) in image pixel coords,
                    where N = feat_h * feat_w * num_anchors.
    """
    sizes = xp.array(anchor_sizes, dtype=DTYPE)          # (S,)
    ratios = xp.array(aspect_ratios, dtype=DTYPE)        # (R,)
    S = sizes.shape[0]
    R = ratios.shape[0]
    A = S * R   # anchors per spatial location

    # base anchor widths/heights for each (size, ratio)
    # ratio = h / w
    sizes2 = sizes ** 2.0
    # widths and heights for each combination (S,R)
    ws = []
    hs = []
    for s2 in sizes2:
        for r in ratios:
            w = xp.sqrt(s2 / r)
            h = xp.sqrt(s2 * r)
            ws.append(w)
            hs.append(h)
    ws = xp.array(ws, dtype=DTYPE)  # (A,)
    hs = xp.array(hs, dtype=DTYPE)  # (A,)

    # grid of centers in image coordinates
    stride = feat_stride
    shifts_x = xp.arange(feat_w, dtype=DTYPE) * stride + stride * 0.5
    shifts_y = xp.arange(feat_h, dtype=DTYPE) * stride + stride * 0.5
    shift_y, shift_x = xp.meshgrid(shifts_y, shifts_x, indexing="ij")  # (H,W)

    shift_x = shift_x.reshape(-1)  # (H*W,)
    shift_y = shift_y.reshape(-1)  # (H*W,)
    num_positions = shift_x.shape[0]

    # expand centers and anchor shapes
    # centers: (N_pos, 1) -> broadcast with (A,)
    center_x = shift_x[:, None]  # (N_pos,1)
    center_y = shift_y[:, None]  # (N_pos,1)

    # anchors dims: (1,A)
    ws = ws.reshape(1, A)
    hs = hs.reshape(1, A)

    x1 = center_x - 0.5 * ws
    y1 = center_y - 0.5 * hs
    x2 = center_x + 0.5 * ws
    y2 = center_y + 0.5 * hs

    anchors = xp.stack([x1, y1, x2, y2], axis=-1)  # (N_pos, A, 4)
    anchors = anchors.reshape(-1, 4)               # (N_pos*A, 4)

    return anchors


def _encode_boxes(self, proposals, gt_boxes):
    """
    Encode ground truth boxes relative to proposals.
    proposals: (N,4) xp array
    gt_boxes:  (N,4) xp array
    Returns: (N,4) deltas (dx,dy,dw,dh)
    """
    px1, py1, px2, py2 = proposals[:, 0], proposals[:, 1], proposals[:, 2], proposals[:, 3]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    pw = xp.maximum(px2 - px1, 1e-6)
    ph = xp.maximum(py2 - py1, 1e-6)
    pcx = px1 + 0.5 * pw
    pcy = py1 + 0.5 * ph

    gw = xp.maximum(gx2 - gx1, 1e-6)
    gh = xp.maximum(gy2 - gy1, 1e-6)
    gcx = gx1 + 0.5 * gw
    gcy = gy1 + 0.5 * gh

    dx = (gcx - pcx) / pw
    dy = (gcy - pcy) / ph
    dw = xp.log(gw / pw)
    dh = xp.log(gh / ph)

    deltas = xp.stack([dx, dy, dw, dh], axis=1)
    return deltas


def _decode_boxes(anchors, deltas):
    """
    Decode box deltas relative to anchors.

    Args:
        anchors: (N, 4) xp.ndarray, (x1, y1, x2, y2)
        deltas:  (N, 4) Tensor or xp.ndarray with (dx, dy, dw, dh)

    Returns:
        boxes: (N, 4) xp.ndarray (x1, y1, x2, y2)
    """
    if isinstance(deltas, Tensor):
        deltas = deltas.to_compute()
    # anchors is already xp.ndarray

    ax1 = anchors[:, 0]
    ay1 = anchors[:, 1]
    ax2 = anchors[:, 2]
    ay2 = anchors[:, 3]

    aw = ax2 - ax1
    ah = ay2 - ay1
    acx = ax1 + 0.5 * aw
    acy = ay1 + 0.5 * ah

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    # predicted centers
    px = dx * aw + acx
    py = dy * ah + acy
    pw = xp.exp(dw) * aw
    ph = xp.exp(dh) * ah

    px1 = px - 0.5 * pw
    py1 = py - 0.5 * ph
    px2 = px + 0.5 * pw
    py2 = py + 0.5 * ph

    boxes = xp.stack([px1, py1, px2, py2], axis=1)
    return boxes


def box_iou_xyxy(boxes1, boxes2, eps=1e-7):
    """
    boxes1: (N,4), boxes2: (M,4)
    Returns IoU matrix (N,M)
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return xp.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=DTYPE)

    b1 = boxes1[:, None, :]  # (N,1,4)
    b2 = boxes2[None, :, :]  # (1,M,4)

    x1 = xp.maximum(b1[..., 0], b2[..., 0])
    y1 = xp.maximum(b1[..., 1], b2[..., 1])
    x2 = xp.minimum(b1[..., 2], b2[..., 2])
    y2 = xp.minimum(b1[..., 3], b2[..., 3])

    inter_w = xp.maximum(x2 - x1, 0.0)
    inter_h = xp.maximum(y2 - y1, 0.0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    area1 = area1[:, None]  # (N,1)
    area2 = area2[None, :]  # (1,M)

    union = area1 + area2 - inter_area
    iou = inter_area / (union + eps)
    return iou


def nms_xyxy(boxes, scores, iou_thresh):
    """
    Vanilla NMS on (x1,y1,x2,y2)

    Args:
        boxes: (N,4) xp.ndarray
        scores: (N,) xp.ndarray
    Returns:
        keep_indices: xp.ndarray of kept indices
    """
    if boxes.shape[0] == 0:
        return xp.array([], dtype=xp.int64)

    order = xp.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        ious = box_iou_xyxy(boxes[i:i+1], boxes[rest])[0]  # (len(rest),)
        mask = ious <= iou_thresh
        order = rest[mask]

    return xp.array(keep, dtype=xp.int64)


def _clip_boxes_to_image(self, boxes, img_h, img_w):
    x1 = xp.clip(boxes[:, 0], 0, img_w - 1)
    y1 = xp.clip(boxes[:, 1], 0, img_h - 1)
    x2 = xp.clip(boxes[:, 2], 0, img_w - 1)
    y2 = xp.clip(boxes[:, 3], 0, img_h - 1)
    return xp.stack([x1, y1, x2, y2], axis=1)


def _filter_small_boxes(self, boxes, min_size):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    ws = x2 - x1
    hs = y2 - y1
    keep = xp.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def _match_proposals_targets(self, proposals_batch, targets, iou_pos=0.5, iou_neg=0.3):
    """
    Very simple matcher:
        - IoU >= iou_pos -> positive (class = gt label)
        - IoU <  iou_neg -> negative (class = 0, background)
        - in (iou_neg, iou_pos) ignored for training

    Args:
        proposals_batch: list of length B, each (N_i,4) xp array
        targets: list of length B, each dict:
                {"boxes": (M_i,4), "labels": (M_i,)}

    Returns:
        all_proposals: (N_total,4) xp array
        all_labels:    (N_total,) xp int64
        all_reg_targets: (N_total,4) xp array, filled only for positives,
                            zeros for others (we'll mask with labels>0)
    """
    all_props = []
    all_labels = []
    all_reg_tgts = []

    for b, props in enumerate(proposals_batch):
        if isinstance(props, Tensor):
            props = props.to_compute()
        boxes_gt = targets[b]["boxes"]
        labels_gt = targets[b]["labels"]
        if isinstance(boxes_gt, Tensor):
            boxes_gt = boxes_gt.to_compute()
        if isinstance(labels_gt, Tensor):
            labels_gt = labels_gt.to_compute()

        N = props.shape[0]
        if boxes_gt.shape[0] == 0 or N == 0:
            # all proposals = background
            all_props.append(props)
            all_labels.append(xp.zeros((N,), dtype=xp.int64))
            all_reg_tgts.append(xp.zeros((N, 4), dtype=DTYPE))
            continue

        ious = self._box_iou_xyxy(props, boxes_gt)  # (N, M_i)
        max_iou = ious.max(axis=1)
        gt_idx = ious.argmax(axis=1)

        labels = xp.zeros((N,), dtype=xp.int64)  # 0 = background
        reg_targets = xp.zeros((N, 4), dtype=DTYPE)

        pos_mask = max_iou >= iou_pos
        neg_mask = max_iou < iou_neg
        # ignore_mask = (~pos_mask) & (~neg_mask)

        # positives: assign class and bbox targets
        pos_indices = xp.where(pos_mask)[0]
        if pos_indices.size > 0:
            gt_ids = gt_idx[pos_indices]
            labels[pos_indices] = labels_gt[gt_ids]  # assumes labels_gt in [1..K]

            gt_boxes_matched = boxes_gt[gt_ids]          # (P,4)
            props_pos = props[pos_indices]                # (P,4)
            deltas = self._encode_boxes(props_pos, gt_boxes_matched)  # (P,4)
            reg_targets[pos_indices] = deltas

        # negatives: labels already 0, reg_targets zeros

        all_props.append(props)
        all_labels.append(labels)
        all_reg_tgts.append(reg_targets)

    all_props = xp.concatenate(all_props, axis=0)       # (N_total, 4)
    all_labels = xp.concatenate(all_labels, axis=0)     # (N_total,)
    all_reg_tgts = xp.concatenate(all_reg_tgts, axis=0) # (N_total, 4)

    return all_props, all_labels, all_reg_tgts


def _rcnn_loss(self, class_logits, bbox_deltas, proposals_batch, targets):
    """
    class_logits: (N_total, num_classes)
    bbox_deltas:  (N_total, num_classes*4)
    proposals_batch: list of (N_i,4) xp arrays
    targets: list[{"boxes":..., "labels":...}]
    """
    # match proposals to GTs
    props_all, labels_all, reg_targets_all = _match_proposals_targets(
        proposals_batch, targets
    )  # props_all (N,4) xp, labels_all (N,), reg_targets_all (N,4)

    # classification loss
    cls_labels = Tensor(labels_all, requires_grad=False)
    cls_loss = self.ce(class_logits, cls_labels)

    # bbox loss only for positives (labels > 0)
    pos_mask = labels_all > 0
    pos_idx = xp.where(pos_mask)[0]
    if pos_idx.size == 0:
        # no positives, just classification loss
        return cls_loss, Tensor(xp.array(0.0, dtype=DTYPE), requires_grad=False)

    # gather positive deltas: bbox_deltas (N, num_classes*4)
    N, C4 = bbox_deltas.shape
    C = self.num_classes
    bbox_deltas = bbox_deltas.to_compute().reshape(N, C, 4)  # (N, C, 4)

    pos_labels = labels_all[pos_idx]  # (P,), each in [1..C-1]
    pos_reg_targets = reg_targets_all[pos_idx]  # (P,4)

    # pick the deltas corresponding to the GT class for each positive
    # indexes: (P,4)
    pred_pos = []
    for i, (idx, cls_id) in enumerate(zip(pos_idx, pos_labels)):
        cls_id = int(cls_id)
        pred_pos.append(bbox_deltas[idx, cls_id])  # (4,)
    pred_pos = xp.stack(pred_pos, axis=0)  # (P,4)

    pred_pos = Tensor(pred_pos, requires_grad=True, dtype=DTYPE)
    target_pos = Tensor(pos_reg_targets, requires_grad=False, dtype=DTYPE)

    reg_loss = self.reg_loss(pred_pos, target_pos)

    return cls_loss, reg_loss


def _rcnn_inference(self,
                    class_logits,
                    bbox_deltas,
                    proposals_batch,
                    image_shapes,
                    score_thresh=0.05,
                    nms_thresh=0.5,
                    max_detections=100):
    """
    Turn head outputs into final detections per image.

    Returns:
        detections: list of dicts:
            {"boxes": (N_i,4), "scores": (N_i,), "labels": (N_i,)}
    """
    # softmax over classes
    # you can also use your Activation("softmax") if you want
    logits_np = class_logits.to_compute()
    logits_max = logits_np.max(axis=1, keepdims=True)
    exp = xp.exp(logits_np - logits_max)
    probs = exp / xp.sum(exp, axis=1, keepdims=True)  # (N, C)

    N, C = probs.shape
    assert C == self.num_classes

    # reshape bbox_deltas to (N, C, 4)
    deltas_np = bbox_deltas.to_compute().reshape(N, C, 4)

    # concatenate all proposals into one big array already in proposals_batch order
    # we need to reconstruct per-image grouping
    props_concat = []
    batch_indices = []
    for b, props in enumerate(proposals_batch):
        if isinstance(props, Tensor):
            props = props.to_compute()
        Nb = props.shape[0]
        if Nb == 0:
            continue
        props_concat.append(props)
        batch_indices.extend([b] * Nb)
    if len(props_concat) == 0:
        return [{"boxes": xp.zeros((0, 4), dtype=DTYPE),
                    "scores": xp.zeros((0,), dtype=DTYPE),
                    "labels": xp.zeros((0,), dtype=xp.int64)} for _ in range(len(proposals_batch))]

    props_concat = xp.concatenate(props_concat, axis=0)  # (N,4)
    batch_indices = xp.array(batch_indices, dtype=xp.int64)  # (N,)

    detections = []
    B = len(proposals_batch)
    for b in range(B):
        # select RoIs belonging to image b
        mask_b = (batch_indices == b)
        idx_b = xp.where(mask_b)[0]
        if idx_b.size == 0:
            detections.append({
                "boxes": xp.zeros((0, 4), dtype=DTYPE),
                "scores": xp.zeros((0,), dtype=DTYPE),
                "labels": xp.zeros((0,), dtype=xp.int64)
            })
            continue

        props_b = props_concat[idx_b]        # (Nb,4)
        probs_b = probs[idx_b]              # (Nb,C)
        deltas_b = deltas_np[idx_b]         # (Nb,C,4)

        boxes_all = []
        scores_all = []
        labels_all = []

        H_img, W_img = image_shapes[b]

        # per-class decode + NMS (skip background = 0)
        for cls_id in range(1, self.num_classes):
            scores_c = probs_b[:, cls_id]  # (Nb,)
            keep_score = scores_c >= score_thresh
            idx_keep = xp.where(keep_score)[0]
            if idx_keep.size == 0:
                continue

            scores_c = scores_c[idx_keep]
            props_c = props_b[idx_keep]
            deltas_c = deltas_b[idx_keep, cls_id, :]  # (Nc,4)

            boxes_c = self._decode_boxes(props_c, deltas_c)  # (Nc,4)

            # clip to image
            x1 = xp.clip(boxes_c[:, 0], 0, W_img - 1)
            y1 = xp.clip(boxes_c[:, 1], 0, H_img - 1)
            x2 = xp.clip(boxes_c[:, 2], 0, W_img - 1)
            y2 = xp.clip(boxes_c[:, 3], 0, H_img - 1)
            boxes_c = xp.stack([x1, y1, x2, y2], axis=1)

            # NMS
            keep = self._nms_xyxy(boxes_c, scores_c, nms_thresh)
            boxes_c = boxes_c[keep]
            scores_c = scores_c[keep]
            labels_c = xp.full((boxes_c.shape[0],), cls_id, dtype=xp.int64)

            boxes_all.append(boxes_c)
            scores_all.append(scores_c)
            labels_all.append(labels_c)

        if len(boxes_all) == 0:
            detections.append({
                "boxes": xp.zeros((0, 4), dtype=DTYPE),
                "scores": xp.zeros((0,), dtype=DTYPE),
                "labels": xp.zeros((0,), dtype=xp.int64)
            })
            continue

        boxes_all = xp.concatenate(boxes_all, axis=0)
        scores_all = xp.concatenate(scores_all, axis=0)
        labels_all = xp.concatenate(labels_all, axis=0)

        # limit to max_detections
        if boxes_all.shape[0] > max_detections:
            order = xp.argsort(scores_all)[::-1][:max_detections]
            boxes_all = boxes_all[order]
            scores_all = scores_all[order]
            labels_all = labels_all[order]

        detections.append({
            "boxes": boxes_all,
            "scores": scores_all,
            "labels": labels_all
        })

    return detections