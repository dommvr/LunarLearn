import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _count_ngrams
from LunarLearn.core import Tensor
from typing import List

xp = backend.xp


class CorpusBLEU(BaseMetric):
    """
    Compute corpus-level BLEU.
    
    preds: list of token lists   (K, *)
    targets: list of list of token lists  (K, R, *)

        Example format:
        preds = [
            ["the", "cat", "is", "cute"],
            ["hello", "world"]
        ]
        
        targets = [
            [["the", "cat", "is", "very", "cute"]],
            [["hello", "beautiful", "world"], ["hi", "world"]]
        ]

    Returns:
        float BLEU score (corpus-level)
    """
    def __init__(self, max_n: int = 4, eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.max_n = max_n
        self.eps = eps

    def compute(self, preds: List[Tensor], targets: List[Tensor], **kwargs):
        # total clipped counts per n
        clipped_counts = xp.zeros(self.max_n)
        total_counts = xp.zeros(self.max_n)

        # total reference length (closest reference for each candidate)
        preds_length = 0
        targets_length = 0

        for pred, tgt_list in zip(preds, targets):
            pred_len = len(pred)
            preds_length += pred_len

            # pick reference whose length is closest to candidate length
            best_tgt_len = min(tgt_list, key=lambda r: abs(len(r) - pred_len))
            targets_length += len(best_tgt_len)

            # aggregate n-gram counts
            for n in range(1, self.max_n + 1):
                pred_counts = _count_ngrams(pred, n)

                # sum of clipped counts
                clipped = 0
                total = 0

                # max reference n-gram counts across all refs
                tgt_max_counts = {}
                for tgt in tgt_list:
                    tgt_counts = _count_ngrams(tgt, n)
                    for ng, count in tgt_counts.items():
                        tgt_max_counts[ng] = max(tgt_max_counts.get(ng, 0), count)

                # compute clipped precision statistics
                for ng, count in pred_counts.items():
                    total += count
                    clipped += min(count, tgt_max_counts.get(ng, 0))

                clipped_counts[n-1] += clipped
                total_counts[n-1] += total

        # modified precisions p_n
        precisions = []
        for clipped, total in zip(clipped_counts, total_counts):
            if total == 0:
                precisions.append(self.eps)  # avoid log(0)
            else:
                precisions.append(clipped / total)

        # geometric mean
        geo_mean = xp.exp(
            sum((1/self.max_n) * xp.log(p) for p in precisions)
        )

        # brevity penalty
        if preds_length < targets_length:
            BP = xp.exp(1 - targets_length / (preds_length + self.eps))
        else:
            BP = 1.0

        return float(BP * geo_mean)