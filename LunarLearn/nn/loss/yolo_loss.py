import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

DTYPE = backend.DTYPE


class YOLOLoss(BaseLoss):
    def __init__(self,
                 anchors_per_scale,
                 strides,
                 num_classes,
                 img_size,
                 lambda_box=1.0,
                 lambda_obj=1.0,
                 lambda_cls=1.0):
        """
        anchors_per_scale: list of len S, each (A,2)
        strides: list of len S (8,16,32)
        img_size: (img_h, img_w)
        """
        self.anchors_per_scale = anchors_per_scale
        self.strides = strides
        self.num_classes = num_classes
        self.img_size = img_size

        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(self, preds: list[Tensor], targets_per_scale: list[Tensor]):
        """
        preds: list of length S, each (B, A*(5+C), H, W)
        targets_per_scale: list of length S, each (B, A, H, W, 5+C)
            [..., 0]   = obj (0/1)
            [..., 1:5] = gt box (x1,y1,x2,y2) in pixels
            [..., 5:]  = class one-hot
        """
        from LunarLearn.train.models.detection.utils import yolo_decode
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0

        for s, pred in enumerate(preds):
            anchors = self.anchors_per_scale[s]
            stride = self.strides[s]
            targets = targets_per_scale[s]  # (B, A, H, W, 5+C)

            B, A, H, W, _ = targets.shape

            # decode predictions
            boxes_pred, obj_pred, cls_pred = yolo_decode(
                pred,
                anchors=anchors,
                num_classes=self.num_classes,
                stride=stride,
                img_size=self.img_size
            )  # boxes_pred: (B, AHW, 4), obj_pred: (B, AHW), cls_pred: (B, AHW, C)

            # flatten targets to match
            targets_flat = targets.reshape(B, A * H * W, 5 + self.num_classes)
            obj_true = targets_flat[..., 0]              # (B, N)
            box_true = targets_flat[..., 1:5]           # (B, N, 4)
            cls_true = targets_flat[..., 5:]            # (B, N, C)

            # objectness mask
            pos_mask = obj_true > 0.5                   # (B, N)

            # --- Box loss (IoU over positives only) ---
            if bool(pos_mask.any()):
                boxes_pred_pos = boxes_pred[pos_mask]
                box_true_pos = box_true[pos_mask]
                box_loss = ops.iou_loss(boxes_pred_pos, box_true_pos)
            else:
                box_loss = Tensor(0.0)

            # --- Objectness loss (BCE over all cells) ---
            # cast types properly
            obj_true_float = obj_true.astype(DTYPE)
            obj_pred_tensor = obj_pred  # already sigmoid outputs
            obj_loss = ops.binary_cross_entropy(obj_pred_tensor, obj_true_float)

            # --- Class loss (BCE over positives only, multi-label-style) ---
            if bool(pos_mask.any()):
                cls_pred_pos = cls_pred[pos_mask]       # (N_pos, C)
                cls_true_pos = cls_true[pos_mask]       # (N_pos, C), one-hot
                cls_loss = ops.binary_cross_entropy(cls_pred_pos, cls_true_pos.astype(DTYPE))
            else:
                cls_loss = Tensor(0.0)

            total_box_loss += box_loss
            total_obj_loss += obj_loss
            total_cls_loss += cls_loss

        total_loss = (self.lambda_box * total_box_loss +
                      self.lambda_obj * total_obj_loss +
                      self.lambda_cls * total_cls_loss)

        total_loss.grad_fn = "yolo_loss"
        total_box_loss.grad_fn = "yolo_loss"
        total_obj_loss.grad_fn = "yolo_loss"
        total_cls_loss.grad_fn = "yolo_loss"
        return total_loss, total_box_loss, total_obj_loss, total_cls_loss