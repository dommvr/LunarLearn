from LunarLearn.nn.optim.schedulers import CosineAnnealing
from LunarLearn.nn.loss import YOLOLoss
from LunarLearn.train.models.detection.utils import build_targets_per_scale


def train_yolo(model,
               dataloader,
               optimizer,
               anchors_per_scale,
               strides,
               num_classes,
               img_size,
               num_epochs,
               device=None):
    """
    model: YOLOv3
    dataloader: yields (images, targets_per_scale)
        images: (B, C, H, W)
        targets_per_scale: list of length S,
            each (B, A, H_s, W_s, 5+num_classes)
    optimizer: your optimizer (e.g. Adam)
    anchors_per_scale: list of len S, each (A, 2)
    strides: list of len S (8,16,32)
    img_size: (img_h, img_w)
    """

    # LR scheduler (epoch-based)
    lr_scheduler = CosineAnnealing(
        target=optimizer,
        attr_name="learning_rate",
        max_epochs=num_epochs,
        min_value=1e-5
    )

    yolo_loss_fn = YOLOLoss(
        anchors_per_scale=anchors_per_scale,
        strides=strides,
        num_classes=num_classes,
        img_size=img_size,
        lambda_box=1.0,
        lambda_obj=1.0,
        lambda_cls=1.0
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # depending on your DataLoader, adjust unpacking:
            images, gt = batch
            targets_per_scale = build_targets_per_scale(gt_batch=gt,
                                                        anchors_per_scale=anchors_per_scale,
                                                        strides=strides,
                                                        num_classes=num_classes,
                                                        img_size=img_size)

            # move to device if you have such logic in your framework
            # images = images.to(device) etc.

            optimizer.zero_grad()

            # forward
            preds = model(images)   # list of S predictions

            # compute loss
            loss, box_loss, obj_loss, cls_loss = yolo_loss_fn(preds, targets_per_scale)

            # backward
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss)
            n_batches += 1

        # step LR scheduler per epoch
        lr_scheduler.step()

        avg_loss = epoch_loss / max(1, n_batches)
        # log something useful
        print(f"Epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f}")

    return model