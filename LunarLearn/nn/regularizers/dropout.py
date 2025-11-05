import LunarLearn.backend as backend
from LunarLearn.tensor import Tensor

xp = backend.xp

def dropout(a: Tensor, keep_prob: float, training: bool = True) -> Tensor:
    """
    Apply dropout regularization to a tensor with autograd support.

    During training, randomly sets elements of the input tensor to zero 
    with probability (1 - keep_prob) and rescales the remaining elements 
    by 1/keep_prob. In inference mode, the input is returned unchanged.

    Parameters
    ----------
    a : Tensor
        Input tensor to apply dropout to.
    keep_prob : float
        Probability of keeping each unit active (0 < keep_prob < 1).
    training : bool, optional
        If True, applies dropout. If False, returns the input unchanged.
        Default is True.

    Returns
    -------
    Tensor
        Tensor with dropout applied in training mode, or unchanged
        in inference mode. Gradients flow correctly through active units.

    Notes
    -----
    - This function is compatible with autograd: gradients flow correctly
      through active units without needing a custom backward.
    - For inference, use `training=False` to bypass dropout.
    """
    if keep_prob <= 0 or keep_prob >= 1:
        raise ValueError("keep_prob must be in the range (0, 1).")

    if not training:
        return a

    # Generate mask
    mask = (xp.random.rand(*a.shape) < keep_prob).astype(a.dtype)
    mask = Tensor(mask, requires_grad=False, dtype=a.dtype)

    # Scaling factor
    scale = Tensor(1.0 / keep_prob, requires_grad=False, dtype=a.dtype)

    return (a * mask) * scale