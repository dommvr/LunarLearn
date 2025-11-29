import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.core import Tensor, ops
from typing import Optional

xp = backend.xp


class Perplexity(BaseMetric):
    def __init__(self, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.eps = eps

    def compute(self, logits: Tensor, targets: Tensor, mask: Optional[Tensor] = None, **kwargs):
        if logits.ndim == 2:
            # (N, V), treat as (N, 1, V)
            N, V = logits.shape
            T = 1
            logits_flat = logits.reshape(N * T, V)
            targets_flat = targets.reshape(N * T)
        elif logits.ndim == 3:
            # (N, T, V)
            N, T, V = logits.shape
            logits_flat = logits.reshape(N * T, V)
            targets_flat = targets.reshape(N * T)
        else:
            raise ValueError("logits must have shape (N, V) or (N, T, V).")

        # compute log probabilities over vocab
        log_probs = ops.log_softmax(logits_flat, axis=-1, epsilon=self.eps)

        # gather log p(y_i)
        idx = ops.arange(logits_flat.shape[0])
        # targets_flat must be int dtype
        target_log_probs = log_probs[idx.data, targets_flat.astype(xp.int64).data]

        # negative log-likelihood per token
        nll = -target_log_probs  # shape (N*T,)

        if mask is not None:
            if mask.ndim == 1:
                mask_flat = mask
            else:
                mask_flat = mask.reshape(-1)

            mask_flat = mask_flat.astype(logits_flat.dtype)

            total_weight = ops.sum(mask_flat) + self.eps
            mean_nll = ops.sum(nll * mask_flat) / total_weight
        else:
            mean_nll = ops.mean(nll)

        ppl = ops.exp(mean_nll)
        return float(ppl.data)