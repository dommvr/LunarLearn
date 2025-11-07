from LunarLearn.core import Tensor

def accuracy(preds: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """
    Compute classification accuracy (binary or multi-class).

    This function automatically detects whether the task is binary
    (preds shape (B, 1)) or multi-class (preds shape (B, C)).

    Args:
        preds (Tensor): Model predictions.
            - Binary: shape (B, 1), typically sigmoid output.
            - Multi-class: shape (B, C), typically softmax output.
        targets (Tensor): Ground-truth labels.
            - Binary: shape (B,) or (B, 1).
            - Multi-class: shape (B,) with class indices or (B, C) one-hot.
        threshold (float, optional): Probability threshold for binary
            classification. Default is 0.5.

    Returns:
        float: Accuracy in [0, 1].
    """
    preds_data = preds.data

    # ----- Binary classification -----
    if preds_data.ndim == 2 and preds_data.shape[1] == 1:
        pred_labels = (preds_data > threshold).astype(int).reshape(-1)
        targets = targets.reshape(-1).astype(int)

    # ----- Multi-class classification -----
    else:
        pred_labels = preds_data.argmax(axis=1)
        if targets.ndim == 2:  # one-hot
            targets = targets.argmax(axis=1)

    correct = (pred_labels == targets).sum()
    return (correct / len(targets)).item()