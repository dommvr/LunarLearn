import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module
from LunarLearn.nn.layers import BaseLayer, Conv2D, Flatten, Dense
from LunarLearn.core import Tensor, ops
from LunarLearn.train.models.detection.utils import _generate_anchors, _decode_boxes, _clip_boxes_to_image, _filter_small_boxes, box_iou_xyxy, nms_xyxy

xp = backend.xp
DTYPE = backend.DTYPE


class RPNHead(BaseLayer):
    def __init__(self,
                 channels: int,
                 num_anchors: int):
        super().__init__(trainable=True)
        self.conv = Conv2D(filters=channels,
                           kernel_size=3,
                           strides=1,
                           padding=1)
        self.objectness = Conv2D(filters=num_anchors,
                                 kernel_size=1,
                                 strides=1)
        self.bbox_deltas = Conv2D(filters=num_anchors*4,
                                  kernel_size=1,
                                  strides=1)
        
    def forward(self, x: Tensor) -> Tensor:
        t = self.conv(x)
        t = ops.relu(t)
        obj_logits = self.objectness(t)
        bbox_deltas = self.bbox_deltas(t)
        return obj_logits, bbox_deltas
    

class RegionProposalNetwork(BaseLayer):
    """
    RPN:
      - takes feature map
      - predicts objectness + box deltas for anchors
      - decodes proposals, applies NMS

    During training:
      return proposals + (rpn_cls_loss, rpn_reg_loss)
    During eval:
      return proposals only
    """
    def __init__(self,
                 in_channels: int,
                 anchor_sizes=(32, 64, 128, 256, 512),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 feat_stride=16,
                 pre_nms_topk=6000,
                 post_nms_topk=1000,
                 nms_thresh=0.7,
                 min_box_size=0):
        super().__init__(trainable=True)

        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.feat_stride = feat_stride
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size

        self.num_anchors = len(anchor_sizes) * len(aspect_ratios)
        self.head = RPNHead(in_channels, self.num_anchors)

    def forward(self,
                feature: Tensor,
                image_shape: tuple[int, int],
                targets=None):
        """
        feature: (B, C, H, W) from backbone/FPN
        image_shape: (H_img, W_img)

        If targets is None: inference mode (return proposals only).
        Otherwise: training mode (also return RPN losses).
        """
        B, C, H, W = feature.shape

        obj_logits, bbox_deltas = self.head(feature)  # (B, A, H, W), (B, 4A, H, W)

        # Flatten per image
        obj_logits_flat = obj_logits.reshape(B, self.num_anchors * H * W)
        bbox_deltas_flat = bbox_deltas.reshape(B, self.num_anchors * H * W, 4)

        # Generate anchors once per feature map size
        anchors = _generate_anchors(H, W)  # (N, 4), N = A * H * W

        proposals_batch = []
        rpn_loss = None

        for b in range(B):
            scores_np = obj_logits_flat[b]          # (N,)
            deltas = bbox_deltas_flat[b]         # (N,4)
            # decode relative to anchors
            boxes = _decode_boxes(anchors, deltas)  # (N,4)

            # Clip to image
            boxes = _clip_boxes_to_image(boxes,
                                              image_shape[0],
                                              image_shape[1])

            # Filter tiny boxes
            keep_small = _filter_small_boxes(boxes, self.min_box_size)
            boxes = boxes[keep_small]
            scores_np = scores_np[keep_small]

            if boxes.shape[0] == 0:
                proposals_batch.append(
                    xp.zeros((0, 4), dtype=DTYPE)
                )
                continue

            # Pre-NMS topk
            if boxes.shape[0] > self.pre_nms_topk:
                order = xp.argsort(scores_np)[::-1][:self.pre_nms_topk]
                boxes = boxes[order]
                scores_np = scores_np[order]

            # NMS
            keep_idx = nms_xyxy(boxes, scores_np, self.nms_thresh)
            boxes = boxes[keep_idx]
            scores_np = scores_np[keep_idx]

            # Post-NMS topk
            if boxes.shape[0] > self.post_nms_topk:
                boxes = boxes[:self.post_nms_topk]
                scores_np = scores_np[:self.post_nms_topk]

            proposals_batch.append(boxes)

        if targets is None:
            # inference: just return proposals, you can later also return scores if you want
            return proposals_batch, rpn_loss
        else:
            # training: you'd compute classification/regression targets & loss here
            # using assignment logic similar in spirit to your YOLO build_targets_per_scale.
            # I'm not inventing your loss API here.
            return proposals_batch, rpn_loss
        

class RoIAlign(BaseLayer):
    """
    Very simple single-level RoIAlign:
    - features: (B, C, H, W)
    - rois: list[ (num_rois_i, 4) ] in (x1, y1, x2, y2) for each image
    - output: list of (num_rois_i, C, out_h, out_w)
    """
    def __init__(self,
                 output_size=(7, 7),
                 spatial_scale=1.0):
        super().__init__(trainable=False)
        self.out_h, self.out_w = output_size
        self.spatial_scale = spatial_scale

    def initialize(self, input_shape):
        self.output_shape = input_shape  # not very meaningful here

    def _roi_align_single(self, feat: Tensor, rois_np):
        """
        feat: (C, H, W) for one image
        rois_np: (N, 4) numpy/cupy array
        returns: (N, C, out_h, out_w)
        """
        C, H, W = feat.shape
        out_h, out_w = self.out_h, self.out_w
        N = rois_np.shape[0]

        out = xp.zeros((N, C, out_h, out_w), dtype=feat.dtype)

        # Very naive implementation: regular grid sampling with bilinear sampling.
        # You can replace with something more efficient later.
        feat_np = feat.data if hasattr(feat, "data") else feat

        for n in range(N):
            x1, y1, x2, y2 = rois_np[n] * self.spatial_scale
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            h = max(y2 - y1, 1e-6)
            w = max(x2 - x1, 1e-6)

            for iy in range(out_h):
                for ix in range(out_w):
                    yy = y1 + (iy + 0.5) * h / out_h
                    xx = x1 + (ix + 0.5) * w / out_w

                    # bilinear sample: you can use your own ops. For now, nearest neighbor:
                    iy0 = int(round(yy))
                    ix0 = int(round(xx))
                    iy0 = max(0, min(H - 1, iy0))
                    ix0 = max(0, min(W - 1, ix0))
                    out[n, :, iy, ix] = feat_np[:, iy0, ix0]

        return Tensor(out, requires_grad=True, dtype=feat.dtype)

    def forward(self, features: Tensor, rois_batch: list[xp.ndarray]):
        """
        features: (B, C, H, W)
        rois_batch: list of (num_rois_i, 4) numpy/cupy arrays
        """
        B, C, H, W = features.shape
        outputs = []
        for b in range(B):
            feat_b = features[b]  # (C, H, W)
            rois_b = rois_batch[b]
            if rois_b.shape[0] == 0:
                # empty
                outputs.append(
                    Tensor(xp.zeros((0, C, self.out_h, self.out_w), dtype=features.dtype),
                           requires_grad=True, dtype=features.dtype)
                )
                continue
            out_b = self._roi_align_single(feat_b, rois_b)
            outputs.append(out_b)
        return outputs
    

class FastRCNNHead(BaseLayer):
    """
    Two-FC head with classification and bbox regression.

    Input:
        roi_feats: (N, C, H, W) (after RoIAlign)
    Output:
        class_logits: (N, num_classes)
        bbox_deltas:  (N, num_classes * 4)
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 pool_h: int = 7,
                 pool_w: int = 7,
                 hidden_dim: int = 1024):
        super().__init__(trainable=True)
        self.in_channels = in_channels
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.num_classes = num_classes

        self.flatten = Flatten()
        self.fc1 = Dense(nodes=hidden_dim, activation="relu")
        self.fc2 = Dense(nodes=hidden_dim, activation="relu")

        self.cls_score = Dense(nodes=num_classes)          # logits
        self.bbox_pred = Dense(nodes=num_classes * 4)      # class-specific boxes

    def forward(self, roi_feats: Tensor):
        # roi_feats: (N, C, H, W)
        x = self.flatten(roi_feats)
        x = self.fc1(x)
        x = self.fc2(x)

        class_logits = self.cls_score(x)   # (N, num_classes)
        bbox_deltas = self.bbox_pred(x)    # (N, num_classes * 4)
        return class_logits, bbox_deltas
    

class FasterRCNN(Module):
    """
    Minimal Faster R-CNN:
      - backbone: returns feature map (B, C, H, W)
      - RPN: proposals per image
      - RoIAlign: fixed-size features per proposal
      - FastRCNN head: classification + box regression

    For training: you pass `targets` & compute losses.
    For inference: targets=None -> returns detections.
    """
    def __init__(self,
                 backbone: Module,
                 backbone_out_channels: int,
                 num_classes: int,
                 anchor_sizes=(32, 64, 128, 256, 512),
                 aspect_ratios=(0.5, 1.0, 2.0),
                 feat_stride=16,
                 pool_size=(7, 7)):
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        self.rpn = RegionProposalNetwork(
            in_channels=backbone_out_channels,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            feat_stride=feat_stride
        )

        self.roi_align = RoIAlign(output_size=pool_size,
                                  spatial_scale=1.0 / feat_stride)

        self.head = FastRCNNHead(
            in_channels=backbone_out_channels,
            num_classes=num_classes,
            pool_h=pool_size[0],
            pool_w=pool_size[1],
            hidden_dim=1024
        )

    def forward(self,
                images: Tensor,
                image_shapes: list[tuple[int, int]],
                targets=None):
        """
        images: (B, 3, H, W) tensor
        image_shapes: list[(H_img, W_img)] per image (after any resize).
        targets: list of dicts (training only), each:
            {
                "boxes":  (N_i, 4) in xyxy pixel coords,
                "labels": (N_i,)    in [1..num_classes-1]
            }

        Returns:
            If targets is None:
                detections: list[{"boxes","scores","labels"}]
            Else:
                losses: {"rpn_loss": Tensor, "rcnn_cls_loss": Tensor, "rcnn_reg_loss": Tensor}
        """
        B = images.shape[0]

        # 1) Backbone
        features = self.backbone(images)   # (B, C, Hf, Wf)

        # 2) RPN: proposals per image
        # for simplicity assume all images have same shape; use first
        rpn_img_shape = image_shapes[0] if isinstance(image_shapes, list) else image_shapes
        proposals_batch, rpn_loss = self.rpn(features, rpn_img_shape, targets=None)
        # rpn_loss is placeholder here; you can later pass targets to train RPN

        # 3) RoIAlign features per proposal
        roi_feats_batch = self.roi_align(features, proposals_batch)
        # flatten into single batch
        roi_batch_concat = []
        for feats_b in roi_feats_batch:
            if feats_b.shape[0] == 0:
                continue
            roi_batch_concat.append(feats_b)

        if len(roi_batch_concat) == 0:
            # no proposals at all
            if targets is None:
                return [{"boxes": xp.zeros((0, 4), dtype=DTYPE),
                            "scores": xp.zeros((0,), dtype=DTYPE),
                            "labels": xp.zeros((0,), dtype=xp.int64)} for _ in range(B)]
            else:
                zero = Tensor(xp.array(0.0, dtype=DTYPE), requires_grad=False)
                return {"rpn_loss": zero,
                        "rcnn_cls_loss": zero,
                        "rcnn_reg_loss": zero}

        roi_batch = ops.concatenate(roi_batch_concat, axis=0)  # (N_total, C, H, W)

        # 4) Fast R-CNN head
        class_logits, bbox_deltas = self.head(roi_batch)  # (N_total,C), (N_total,C*4)

        if targets is None:
            # Inference mode
            detections = self._rcnn_inference(
                class_logits,
                bbox_deltas,
                proposals_batch,
                image_shapes,
                score_thresh=0.05,
                nms_thresh=0.5,
                max_detections=100
            )
            return detections
        else:
            # Training mode: only RCNN losses here, RPN loss left as TODO
            rcnn_cls_loss, rcnn_reg_loss = self._rcnn_loss(
                class_logits,
                bbox_deltas,
                proposals_batch,
                targets
            )

            if rpn_loss is None:
                rpn_loss = Tensor(xp.array(0.0, dtype=DTYPE), requires_grad=False)

            return {
                "rpn_loss": rpn_loss,
                "rcnn_cls_loss": rcnn_cls_loss,
                "rcnn_reg_loss": rcnn_reg_loss
            }