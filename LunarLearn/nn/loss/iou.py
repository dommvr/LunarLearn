from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, ops

class IoU(BaseLoss):
    def forward(self, preds: Tensor, targets: Tensor, eps: float = 1e-7) -> Tensor:
        return ops.iou_loss(preds, targets, eps=eps)