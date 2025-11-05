import LunarLearn.backend as backend
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer
from LunarLearn.tensor import ops
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class FisherLogitTraceRegularizer(BaseRegularizer):
    """
    Fisher (logit) trace proxy regularizer.

    Penalizes sum_k p_k(1 - p_k) per sample (trace of Fisher w.r.t. logits).
    Requires only forward probabilities (e.g., after softmax or sigmoid).
    Does NOT require higher-order gradients.

    Args:
        lam (float): Strength of the regularization.
        attr (str): Name of the activation attribute on the last layer to read
                    probabilities from. Default 'A' (common in your layers).
        reduce (str): 'mean' or 'sum' over batch.
    """
    def __init__(self, lam: float = 1e-4, attr: str = "A", reduce: str = "mean"):
        self.lam = lam
        self.attr = attr
        if reduce not in ("mean", "sum"):
            raise ValueError("reduce must be 'mean' or 'sum'")
        self.reduce = reduce

    def __call__(self, model):
        """
        Compute model-level Fisher proxy by inspecting the last layer outputs.

        Returns:
            Tensor: scalar regularization term (autograd-compatible).
        """
        # Try to find the last leaf-like module that has an activation tensor
        last_layer = None
        for name, attr in model.__dict__.items():
            # heuristic: pick the last thing with attribute 'A'
            if hasattr(attr, self.attr):
                last_layer = attr

        if last_layer is None:
            # Fallback: try model-level 'A' (if you stash it on the model)
            if hasattr(model, self.attr):
                probs = getattr(model, self.attr)
            else:
                # Nothing to regularize against
                return Tensor(0.0)
        else:
            probs = getattr(last_layer, self.attr, None)
            if probs is None:
                return Tensor(0.0)

        # probs expected shape (N, C) or (N, 1) after sigmoid/softmax
        # fisher trace per sample: sum p(1-p)
        one = Tensor(1.0, requires_grad=False, dtype=probs.dtype)
        per_sample = ops.sum(probs * (one - probs), axis=-1)  # (N,)

        if self.reduce == "mean":
            reg = ops.mean(per_sample)
        else:
            reg = ops.sum(per_sample)

        return reg * self.lam

    # NOTE: we do not override .loss(param) here intentionally — this
    # regularizer is about *outputs*, not per-parameter penalties.