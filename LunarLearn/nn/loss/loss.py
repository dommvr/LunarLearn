import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor, ops

DTYPE = backend.DTYPE


class BaseLoss:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class BinaryCrossEntropyDice(BaseLoss):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.binary_cross_entropy_dice(predictions, targets)
   

class BinaryCrossEntropyWithLogits(BaseLoss):
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.binary_cross_entropy_with_logits(predictions, targets)
  

class BinaryCrossEntropy(BaseLoss):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.binary_cross_entropy(predictions, targets, epsilon=self.eps)


class CosineSimilarity(BaseLoss):
    """
    Cosine similarity loss with autograd support.

    Computes 1 - cosine similarity between predicted vectors and target vectors.
    Fully compatible with autograd: gradients flow through normalization and
    vector operations.

    Args:
        None (all parameters are passed to `forward`).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the mean 1 - cosine similarity loss over a batch.

            Args:
                predictions (Tensor): Predicted vectors of shape (B, D).
                targets (Tensor): Target vectors of shape (B, D).
                epsilon (float, optional): Small constant to avoid division by zero. Default 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean loss. Gradients are tracked automatically.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.cosine_similarity(predictions, targets, epsilon=self.eps)


class CrossEntropy(BaseLoss):
    """
    Cross-entropy loss with autograd support.

    Computes the mean cross-entropy between predicted probabilities and target labels.
    Fully compatible with autograd: gradients flow through log, multiplication, and mean.

    Args:
        None (all parameters are passed to `forward`).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the mean cross-entropy loss.

            Args:
                predictions (Tensor): Predicted probabilities (softmax output) of shape (B, C).
                targets (Tensor): Target labels. Either integer indices (B,) or one-hot (B, C).
                epsilon (float, optional): Small constant to avoid log(0). Default 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean loss. Gradients are tracked automatically.
    """
    def __init__(self, axis: int = -1, eps: float = 1e-8):
        self.axis = axis
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.cross_entropy(predictions, targets, axis=self.axis, epsilon=self.eps)


class Dice(BaseLoss):
    def __init__(self, smooth: float = 1.0):
        self.smooth = smooth

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.dice(predictions, targets, smooth=self.smooth)


class Focal(BaseLoss):
    """
    Focal loss with autograd support.

    This loss is designed to address class imbalance by down-weighting well-classified examples.
    It extends the standard cross-entropy loss by a modulating factor that reduces the loss for
    easy examples.

    Args:
        alpha (float, optional): Weighting factor for the rare class. Default is 1.0.
        gamma (float, optional): Focusing parameter that reduces loss contribution from easy examples. Default is 2.0.

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the focal loss over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted probabilities (softmax output) of shape (B, C).
                targets (Tensor): Target labels. Either integer indices (B,) or one-hot (B, C).
                epsilon (float, optional): Small constant to avoid log(0). Default is 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean focal loss. Gradients are tracked automatically.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__(trainable=False)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.focal(predictions, targets, alpha=self.alpha, gamma=self.gamma)


class Huber(BaseLoss):
    """
    Huber loss with autograd support.

    The Huber loss is less sensitive to outliers than mean squared error (MSE). 
    It behaves like MSE for small errors and like mean absolute error (MAE) for 
    large errors, with a transition defined by the delta threshold.

    Args:
        delta (float, optional): Threshold at which the loss switches from quadratic 
            to linear. Default is 1.0.

    Methods:
        forward(predictions: Tensor, targets: Tensor, delta: float = 1.0) -> Tensor:
            Computes the Huber loss over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted values of shape (B, D) or (B, C).
                targets (Tensor): Target values or labels. Can be integer indices, one-hot,
                    or continuous values depending on the task.
                delta (float, optional): Threshold at which to switch from quadratic to linear loss.
                    Default is 1.0.

            Returns:
                Tensor: Scalar tensor containing the mean Huber loss. Gradients are tracked automatically.
    """
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.huber(predictions, targets, delta=self.delta)


class IoU(BaseLoss):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        return ops.iou_loss(preds, targets, eps=self.eps)


class KLDivergence(BaseLoss):
    """
    Kullback-Leibler (KL) Divergence loss with autograd support.

    This loss measures the divergence between two probability distributions. 
    It is often used when comparing a predicted probability distribution to 
    a target distribution (e.g., soft labels or teacher-student models).

    Methods:
        forward(predictions: Tensor, targets: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the KL divergence over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted probabilities (softmax output) of shape (B, C).
                targets (Tensor): Target probability distributions of shape (B, C).
                epsilon (float, optional): Small constant to avoid log(0). Default is 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean KL divergence. Gradients are tracked automatically.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.kl_divergence(predictions, targets, epsilon=self.eps)


class MeanAbsoluteError(BaseLoss):
    """
    Mean Absolute Error (MAE) loss with autograd support.

    This loss computes the average absolute difference between predictions and targets.
    It supports integer class labels (converted to one-hot) or direct target values.

    Methods:
        forward(predictions: Tensor, targets: Tensor) -> Tensor:
            Computes the MAE over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted values of shape (B, C) or (B, 1).
                targets (Tensor): Target values. Either integer class indices (B,) 
                                  or one-hot / continuous targets (B, C).

            Returns:
                Tensor: Scalar tensor containing the mean absolute error. 
                        Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.mean_absolute_error(predictions, targets)


class MeanSquaredError(BaseLoss):
    """
    Mean Squared Error (MSE) loss with autograd support.

    This loss computes the average squared difference between predictions and targets.
    It supports integer class labels (converted to one-hot) or direct target values.

    Methods:
        forward(predictions: Tensor, targets: Tensor) -> Tensor:
            Computes the MSE over a batch of predictions and targets.

            Args:
                predictions (Tensor): Predicted values of shape (B, C) or (B, 1).
                targets (Tensor): Target values. Either integer class indices (B,) 
                                or one-hot / continuous targets (B, C).

            Returns:
                Tensor: Scalar tensor containing the mean squared error. 
                        Gradients are tracked automatically.
    """
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return ops.mean_squared_error(predictions, targets)


class Triplet(BaseLoss):
    """
    Triplet loss with autograd support.

    This loss encourages the distance between an anchor and a positive sample 
    (same class) to be smaller than the distance between the anchor and a negative 
    sample (different class) by at least a margin. It is commonly used in 
    metric learning and embedding models.

    Methods:
        forward(anchor: Tensor, positive: Tensor, negative: Tensor, epsilon: float = 1e-15) -> Tensor:
            Computes the triplet loss over a batch of embeddings.

            Args:
                anchor (Tensor): Anchor embeddings of shape (B, D).
                positive (Tensor): Positive embeddings (same class as anchor), shape (B, D).
                negative (Tensor): Negative embeddings (different class), shape (B, D).
                epsilon (float, optional): Small constant for numerical stability. 
                                        Default is 1e-15.

            Returns:
                Tensor: Scalar tensor containing the mean triplet loss. 
                        Gradients are tracked automatically.
    """
    def __init__(self, margin: float = 0.2, eps: float = 1e-8):
        self.margin = margin
        self.eps = eps

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return ops.triplet(anchor, positive, negative, margin=self.margin, epsilon=self.eps)


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
