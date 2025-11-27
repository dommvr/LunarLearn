import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor, ops
from typing import Optional, List, Tuple

xp = backend.xp


def true_positive(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class TP
        per_class_tp = xp.sum((preds_bin == 1) & (targets == 1), axis=0)
        # total TP
        total_tp = int(xp.sum(per_class_tp))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_tp = xp.zeros(C, dtype=int)
        # TP per class: prediction must match and equal the class
        matches = (preds == targets)
        for c in range(C):
            per_class_tp[c] = xp.sum(matches & (targets == c))
        
        total_tp = int(xp.sum(per_class_tp))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_tp, per_class_tp


def true_negative(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class TN
        per_class_tn = xp.sum((preds_bin == 0) & (targets == 0), axis=0)
        # total TN
        total_tn = int(xp.sum(per_class_tn))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_tn = xp.zeros(C, dtype=int)
        # TN per class: model did NOT predict class c AND true label is NOT class c
        for c in range(C):
            per_class_tn[c] = xp.sum((preds != c) & (targets != c))
        
        total_tn = int(xp.sum(per_class_tn))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_tn, per_class_tn


def false_positive(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class FP
        per_class_fp = xp.sum((preds_bin == 1) & (targets == 0), axis=0)
        # total FP
        total_fp = int(xp.sum(per_class_fp))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_fp = xp.zeros(C, dtype=int)
        # FP per class: model did predict class c AND true label is NOT class c
        for c in range(C):
            per_class_fp[c] = xp.sum((preds == c) & (targets != c))
        
        total_fp = int(xp.sum(per_class_fp))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_fp, per_class_fp


def false_negative(preds: Tensor, targets: Tensor, threshold: float = 0.5):
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # infer classes from width
        C = targets.shape[1]
        # threshold predictions if not already binary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        # per-class FN
        per_class_fn = xp.sum((preds_bin == 0) & (targets == 1), axis=0)
        # total FN
        total_fn = int(xp.sum(per_class_fn))

    # MULTI-CLASS or BINARY with shape (N,)
    elif preds.ndim == 1:
        # get number of classes
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_fn = xp.zeros(C, dtype=int)
        # FN per class: model did NOT predict class c AND true label is class c
        for c in range(C):
            per_class_fn[c] = xp.sum((preds != c) & (targets == c))
        
        total_fn = int(xp.sum(per_class_fn))
    
    else:
        raise ValueError("Unsupported shape for preds or targets.")
    
    return total_fn, per_class_fn


def accuracy(preds: Tensor, targets: Tensor, threshold: float = 0.5, eps: float = 1e-12):
    """
    Accuracy for:
      - binary
      - multi-label (micro/macro)
      - multi-class

    Returns:
        micro_accuracy
        macro_accuracy
        weighted_accuracy
        per_class_accuracy
    """
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        # threshold if necessary
        if preds.dtype not in [xp.int32, xp.int64, xp.uint8]:
            preds_bin = (preds > threshold).astype(int)
        else:
            preds_bin = preds

        correct = (preds_bin == targets)

        # per-class accuracy
        per_class_accuracy = xp.mean(correct, axis=0)

        # micro accuracy (flatten everything)
        micro_accuracy = float(xp.mean(correct))

        # macro accuracy (mean over classes)
        macro_accuracy = float(xp.mean(per_class_accuracy))

        # weighted accuracy (weight by support)
        support = xp.sum(targets == 1, axis=0)
        total_support = xp.sum(support) + eps
        weighted_accuracy = float(xp.sum(per_class_accuracy * support) / total_support)

        return micro_accuracy, macro_accuracy, weighted_accuracy, per_class_accuracy

    # MULTI-CLASS or BINARY (N,)
    elif preds.ndim == 1:
        # per-sample correctness
        correct = preds == targets
        micro_accuracy = float(xp.mean(correct))

        # per-class accuracy
        C = int(xp.max(preds.max(), targets.max()) + 1)
        per_class_accuracy = xp.zeros(C)

        for c in range(C):
            mask = (targets == c)
            if xp.sum(mask) == 0:
                per_class_accuracy[c] = 0.0
            else:
                per_class_accuracy[c] = float(xp.mean(correct[mask]))

        macro_accuracy = float(xp.mean(per_class_accuracy))

        # weighted by class frequency
        support = xp.array([xp.sum(targets == c) for c in range(C)], dtype=float)
        total_support = xp.sum(support) + eps
        weighted_accuracy = float(xp.sum(per_class_accuracy * support) / total_support)

        return micro_accuracy, macro_accuracy, weighted_accuracy, per_class_accuracy

    else:
        raise ValueError("Unsupported shape for preds or targets.")


def precision(preds: Tensor, targets: Tensor, threshold: float = 0.5, eps: float = 1e-12):
    """
    Universal precision for:
      - binary
      - multi-class
      - multi-label

    Returns:
        micro_precision: float
        macro_precision: float
        weighted_precision: float
        per_class_precision: array (C,)
    """
    # true positives and false positives
    total_tp, per_class_tp = true_positive(preds, targets, threshold)
    total_fp, per_class_fp = false_positive(preds, targets, threshold)

    # avoid division by zero
    per_class_precision = per_class_tp / (per_class_tp + per_class_fp + eps)

    # micro precision: TP / (TP + FP)
    micro_precision = float(total_tp / (total_tp + total_fp + eps))

    # macro precision: mean over classes
    macro_precision = float(xp.mean(per_class_precision))

    # weighted precision: weight by support (true occurrences of each class)
    # Works for multi-class and multi-label.
    # For multi-label: support = number of positives for that label.
    # For multi-class: support = count of samples belonging to class c.
    targets_data = targets.data
    if targets_data.ndim == 2:
        support = xp.sum(targets_data == 1, axis=0)
    else:
        C = len(per_class_tp)
        support = xp.array([xp.sum(targets_data == c) for c in range(C)])

    total_support = xp.sum(support) + eps
    weighted_precision = float(xp.sum(per_class_precision * support) / total_support)

    return micro_precision, macro_precision, weighted_precision, per_class_precision


def recall(preds: Tensor, targets: Tensor, threshold: float = 0.5, eps: float = 1e-12):
    """
    Universal recall for:
      - binary
      - multi-class
      - multi-label

    Returns:
        micro_recall: float
        macro_recall: float
        weighted_recall: float
        per_class_recall: array (C,)
    """
    # true positives and false negative
    total_tp, per_class_tp = true_positive(preds, targets, threshold)
    total_fn, per_class_fn = false_negative(preds, targets, threshold)

    # avoid division by zero
    per_class_recall = per_class_tp / (per_class_tp + per_class_fn + eps)

    # micro recall: TP / (TP + FN)
    micro_recall = float(total_tp / (total_tp + total_fn + eps))

    # macro recall: mean over classes
    macro_recall = float(xp.mean(per_class_recall))

    # weighted recall: weight by support (true occurrences of each class)
    # Works for multi-class and multi-label.
    # For multi-label: support = number of positives for that label.
    # For multi-class: support = count of samples belonging to class c.
    targets_data = targets.data
    if targets_data.ndim == 2:
        support = xp.sum(targets_data == 1, axis=0)
    else:
        C = len(per_class_tp)
        support = xp.array([xp.sum(targets_data == c) for c in range(C)])

    total_support = xp.sum(support) + eps
    weighted_recall = float(xp.sum(per_class_recall * support) / total_support)

    return micro_recall, macro_recall, weighted_recall, per_class_recall


def f1_score(preds: Tensor, targets: Tensor, threshold: float = 0.5, eps: float = 1e-12):
    micro_precision, macro_precision, weighted_precision, per_class_precision = precision(preds, targets, threshold, eps)
    micro_recall, macro_recall, weighted_recall, per_class_recall = recall(preds, targets, threshold, eps)

    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall + eps)
    weighted_f1 = 2 * weighted_precision * weighted_recall / (weighted_precision + weighted_recall + eps)
    per_class_f1 = 2 * per_class_precision * per_class_recall / (per_class_precision + per_class_recall + eps)

    return micro_f1, macro_f1, weighted_f1, per_class_f1


def topk_accuracy(preds: Tensor, targets: Tensor, k: int = 5):
    """
    Top-k accuracy for multi-class predictions.

    preds: (N, C) raw logits or probabilities
    targets: (N,) integer class labels
    """
    preds = preds.data
    targets = targets.data

    if preds.ndim != 2:
        raise ValueError("topk_accuracy requires prediction matrix of shape (N, C).")

    # sort descending, take top-k indices
    topk = xp.argsort(-preds, axis=1)[:, :k]

    # check whether true label is in the row's top-k
    correct = xp.any(topk == targets[:, None], axis=1)

    return float(xp.mean(correct))


def _binary_auroc(scores, labels, eps=1e-12):
    # Sort by predicted score
    order = xp.argsort(scores)
    scores = scores[order]
    labels = labels[order]

    # Count positives and negatives
    P = xp.sum(labels == 1)
    N = xp.sum(labels == 0)

    if P == 0 or N == 0:
        return 0.5  # undefined, neutral score

    # Compute rank-sum AUROC (Mann–Whitney)
    # AUROC = (sum of ranks of positive - P*(P+1)/2) / (P*N)
    ranks = xp.arange(1, len(scores) + 1)
    pos_ranks = xp.sum(ranks[labels == 1])

    auc = (pos_ranks - P * (P + 1) / 2) / (P * N + eps)
    return float(auc)


def auroc(preds: Tensor, targets: Tensor, eps: float = 1e-12):
    """
    AUROC for:
      - binary (N, 1) or (N,) with 0/1 labels
      - multi-label (N, C) with 0/1 targets per class
      - multi-class one-vs-rest *if* targets are already binarized to (N, C)

    Returns:
        micro_auroc: float
        macro_auroc: float
        weighted_auroc: float
        per_class_auroc: array (C,)
    """
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL / BINARY in (N, C) form
    if preds.ndim == 2:
        C = preds.shape[1]
        per_class_auroc = xp.zeros(C, dtype=xp.float32)

        for c in range(C):
            per_class_auroc[c] = _binary_auroc(preds[:, c], targets[:, c], eps)

        # micro AUROC: flatten all scores and labels
        micro_auroc = _binary_auroc(preds.ravel(), targets.ravel(), eps)

        # macro AUROC: unweighted mean over classes
        macro_auroc = float(xp.mean(per_class_auroc))

        # weighted AUROC: weight by number of positives per class
        support = xp.sum(targets == 1, axis=0)          # shape (C,)
        total_support = xp.sum(support) + eps
        weighted_auroc = float(xp.sum(per_class_auroc * support) / total_support)

        return micro_auroc, macro_auroc, weighted_auroc, per_class_auroc

    # preds is (N,) → you gave class indices, not scores
    elif preds.ndim == 1:
        raise ValueError("AUROC requires probability scores per class, not class indices.")

    else:
        raise ValueError("Unsupported shape for preds or targets.")


def _binary_auprc(scores, labels, eps=1e-12):
    """
    Compute AUPRC (average precision) for a binary problem.
    scores: (N,) predicted probabilities or logits
    labels: (N,) ground truth 0/1
    """

    # sort by descending score
    order = xp.argsort(-scores)
    scores = scores[order]
    labels = labels[order]

    # cumulative true positives and false positives
    tp = xp.cumsum(labels == 1)
    fp = xp.cumsum(labels == 0)

    P = xp.sum(labels == 1)
    if P == 0:
        return 0.0

    precision = tp / (tp + fp + eps)
    recall = tp / (P + eps)

    # trapezoidal integration under PR curve
    return float(xp.trapz(precision, recall))


def auprc(preds: Tensor, targets: Tensor, eps: float = 1e-12):
    """
    Compute AUPRC (average precision) for:
      - binary classification
      - multi-label classification
      - multi-class classification (one-vs-rest)

    Returns:
        micro_auprc: float
        macro_auprc: float
        weighted_auprc: float
        per_class_auprc: array (C,)
    """
    preds = preds.data
    targets = targets.data

    # MULTI-LABEL or BINARY (N, C)
    if preds.ndim == 2:
        C = preds.shape[1]
        per_class_auprc = xp.zeros(C)

        for c in range(C):
            per_class_auprc[c] = _binary_auprc(preds[:, c], targets[:, c], eps)

        # micro AUPRC: flatten everything (sklearn style)
        micro_auprc = _binary_auprc(preds.ravel(), targets.ravel(), eps)

        # macro AUPRC: unweighted mean
        macro_auprc = float(xp.mean(per_class_auprc))

        # weighted AUPRC: weight by class support
        # support = number of positive samples for each class
        support = xp.sum(targets == 1, axis=0)
        total_support = xp.sum(support) + eps
        weighted_auprc = float(xp.sum(per_class_auprc * support) / total_support)

        return micro_auprc, macro_auprc, weighted_auprc, per_class_auprc

    # MULTI-CLASS (N,)
    elif preds.ndim == 1:
        raise ValueError("AUPRC requires probabilities per class, not class indices.")

    else:
        raise ValueError("Unsupported shape for preds or targets.")


def mse(preds: Tensor, targets: Tensor):
    return ops.mean_squared_error(preds, targets).data


def rmse(preds: Tensor, targets: Tensor):
    return ops.sqrt(ops.mean_squared_error(preds, targets)).data


def mae(preds: Tensor, targets: Tensor):
    return ops.mean_absolute_error(preds, targets).data


def r2_score(preds: Tensor, targets: Tensor, eps: float = 1e-12):
    preds = preds.data
    targets = targets.data

    SS_res = xp.sum((targets - preds)**2)
    SS_tot = xp.sum((targets - xp.mean(targets))**2)
    R2 = float(1 - SS_res / (SS_tot + eps))
    return R2


def _count_ngrams(tokens: Tensor, n: int):
    counts = {}
    for i in range(len(tokens) - n + 1):
        ngram = tuple(int(x) for x in tokens[i:i+n])
        counts[ngram] = counts.get(ngram, 0) + 1
    return counts


def _modified_precision(preds: Tensor, targets: Tensor, n: int):
    preds_ngrams = _count_ngrams(preds, n)
    tgt_ngrams = _count_ngrams(targets, n)

    clipped = 0
    total = 0

    for ng, count in preds_ngrams.items():
        total += count
        clipped += min(count, tgt_ngrams.get(ng, 0))

    if total == 0:
        return 0.0

    return clipped / total


def bleu(preds: Tensor, targets: Tensor, max_n: int = 4, eps: float = 1e-12):
    """
    BLEU score for a single candidate-reference pair.
    preds: list[str] or list[int]
    targets: list[str] or list[int]
    max_n: up to BLEU-N (default BLEU-4)
    """
    # Modified precisions p1..pN
    precisions = []
    for n in range(1, max_n + 1):
        precisions.append(_modified_precision(preds, targets, n))

    # If any precision is zero, BLEU would collapse to zero.
    # We epsilon it to avoid log(0).
    precisions = [p if p > 0 else eps for p in precisions]

    # Geometric mean of the precisions
    log_sum = sum((1 / max_n) * xp.log(p) for p in precisions)
    geo_mean = xp.exp(log_sum)

    # Brevity penalty
    c = len(preds)
    r = len(targets)

    if c == 0:
        return 0.0

    if c < r:
        BP = xp.exp(1 - r / (c + eps))
    else:
        BP = 1.0

    return float(BP * geo_mean)


def corpus_bleu(preds: list[Tensor], targets: list[Tensor], max_n=4, eps=1e-12):
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
    # total clipped counts per n
    clipped_counts = xp.zeros(max_n)
    total_counts = xp.zeros(max_n)

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
        for n in range(1, max_n + 1):
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
            precisions.append(eps)  # avoid log(0)
        else:
            precisions.append(clipped / total)

    # geometric mean
    geo_mean = xp.exp(
        sum((1/max_n) * xp.log(p) for p in precisions)
    )

    # brevity penalty
    if preds_length < targets_length:
        BP = xp.exp(1 - targets_length / (preds_length + eps))
    else:
        BP = 1.0

    return float(BP * geo_mean)


def _lcs_length(a: Tensor, b: Tensor):
    """
    Compute length of LCS between token lists a and b.
    Classic DP O(n*m). Fine for sentence-level use.
    """
    n = len(a)
    m = len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for j in range(1, m+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[n][m]


def rouge_l(preds: Tensor, targets: Tensor, beta: float = 1.2, eps: float = 1e-12):
    """
    Compute ROUGE-L (F-measure form) for one candidate-reference pair.
    """
    lcs = _lcs_length(preds, targets)

    m = len(preds)
    n = len(targets)

    if m == 0 or n == 0:
        return 0.0

    # LCS-based precision & recall
    R_lcs = lcs / (n + eps)
    P_lcs = lcs / (m + eps)

    # F-measure with β
    num = (1 + beta**2) * R_lcs * P_lcs
    den = R_lcs + beta**2 * P_lcs + eps

    return float(num / den)


def corpus_rouge_l(preds: list[Tensor], targets: list[Tensor], beta: float = 1.2, eps: float = 1e-12):
    """
    Corpus-level ROUGE-L.

    preds: list of token lists (K, *)
    targets: list of token lists (K, *)  
                (if you want multiple targets per sample, see below)

    Returns:
        mean ROUGE-L over dataset
    """
    scores = []
    for pred, tgt in zip(preds, targets):
        scores.append(rouge_l(pred, tgt, beta, eps))

    return float(sum(scores) / (len(scores) + eps))


def corpus_rouge_l_multi_ref(preds: list[Tensor], list_of_tgt_lists: list[list[Tensor]], beta: float = 1.2, eps: float = 1e-12):
    """
    Corpus-level ROUGE-L with multiple references per item.

    preds:        (K, *)
    list_of_tgt_lists: (K, R, *)

    For each candidate, score all R references and take max.
    """
    scores = []
    for cand, refs in zip(preds, list_of_tgt_lists):
        best = 0.0
        for ref in refs:
            score = rouge_l(cand, ref, beta, eps)
            if score > best:
                best = score
        scores.append(best)

    return float(sum(scores) / (len(scores) + eps))


def perplexity(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> float:
    """
    Perplexity for language modeling.

    Args:
        logits: Tensor of shape (N, V) or (N, T, V)
                raw unnormalized scores over vocabulary.
        targets: Tensor of shape (N,) or (N, T)
                 integer token IDs (0..V-1).
        mask: optional Tensor of shape (N,) or (N, T)
              1 for valid tokens, 0 for padding.
        eps: small constant for numerical stability.

    Returns:
        Scalar float perplexity.
    """

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
    log_probs = ops.log_softmax(logits_flat, axis=-1, epsilon=eps)

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

        total_weight = ops.sum(mask_flat) + eps
        mean_nll = ops.sum(nll * mask_flat) / total_weight
    else:
        mean_nll = ops.mean(nll)

    ppl = ops.exp(mean_nll)
    return float(ppl.data)


def iou(preds: Tensor, targets: Tensor, threshold: float = 0.5, eps: float = 1e-12):
    """
    Intersection over Union (IoU) for segmentation.

    Supports:
      - preds: (N, C, H, W), targets: (N, H, W)       → multi-class logits + class indices
      - preds: (N, C, H, W), targets: (N, C, H, W)    → multi-label / one-hot masks
      - preds: (N, H, W),    targets: (N, H, W)       → binary segmentation

    Returns:
        micro_iou: float
        macro_iou: float
        weighted_iou: float
        per_class_iou: xp.ndarray of shape (C,)
    """
    p = preds.data
    t = targets.data

    # ---------- CASE 1: logits (N, C, H, W) + class indices (N, H, W) ----------
    if p.ndim == 4 and t.ndim == 3:
        N, C, H, W = p.shape

        # argmax over class dimension -> predicted labels
        pred_labels = xp.argmax(p, axis=1)   # (N, H, W)
        true_labels = t

        # flatten
        pred_flat = pred_labels.reshape(-1)
        true_flat = true_labels.reshape(-1)

        # number of classes
        C = int(max(pred_flat.max(), true_flat.max()) + 1)

        inter = xp.zeros(C, dtype=xp.int64)
        union = xp.zeros(C, dtype=xp.int64)
        support = xp.zeros(C, dtype=xp.int64)   # number of GT pixels per class

        for c in range(C):
            pred_c = (pred_flat == c)
            true_c = (true_flat == c)

            inter[c] = xp.sum(pred_c & true_c)
            union[c] = xp.sum(pred_c | true_c)
            support[c] = xp.sum(true_c)

        per_class_iou = inter / (union + eps)
        micro_iou = float(xp.sum(inter) / (xp.sum(union) + eps))
        macro_iou = float(xp.mean(per_class_iou))

        total_support = xp.sum(support) + eps
        weighted_iou = float(xp.sum(per_class_iou * support) / total_support)

        return micro_iou, macro_iou, weighted_iou, per_class_iou

    # ---------- CASE 2: masks with same shape ---------- 
    # (N, C, H, W) multi-label or (N, H, W) binary
    if p.shape != t.shape:
        raise ValueError("For mask IoU, preds and targets must have the same shape or (N,C,H,W)/(N,H,W).")

    # If not integer, threshold predictions
    if p.dtype not in [xp.int32, xp.int64, xp.uint8]:
        preds_bin = (p > threshold).astype(xp.int32)
    else:
        preds_bin = p

    # Targets: treat any non-zero as 1
    if t.dtype not in [xp.int32, xp.int64, xp.uint8]:
        targets_bin = (t > 0.5).astype(xp.int32)
    else:
        targets_bin = (t != 0).astype(xp.int32)

    if preds_bin.ndim == 3:
        # (N, H, W) → single "class"
        preds_flat = preds_bin.reshape(1, -1)
        targets_flat = targets_bin.reshape(1, -1)
        C = 1
    elif preds_bin.ndim == 4:
        # (N, C, H, W) → multi-label / multi-class one-hot
        N, C, H, W = preds_bin.shape
        preds_flat = preds_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
        targets_flat = targets_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
    else:
        raise ValueError("Unsupported mask shape for IoU: expected (N,H,W) or (N,C,H,W).")

    # intersection & union per class
    intersection = xp.sum((preds_flat == 1) & (targets_flat == 1), axis=1)
    union = xp.sum((preds_flat == 1) | (targets_flat == 1), axis=1)
    support = xp.sum(targets_flat == 1, axis=1)

    per_class_iou = intersection / (union + eps)
    micro_iou = float(xp.sum(intersection) / (xp.sum(union) + eps))
    macro_iou = float(xp.mean(per_class_iou))

    total_support = xp.sum(support) + eps
    weighted_iou = float(xp.sum(per_class_iou * support) / total_support)

    return micro_iou, macro_iou, weighted_iou, per_class_iou


def box_iou(boxes1, boxes2, eps: float = 1e-12):
    """
    Pairwise IoU between two sets of boxes.
    boxes1: Tensor or xp.ndarray, shape (N, 4)
    boxes2: Tensor or xp.ndarray, shape (M, 4)

    Boxes format: [x1, y1, x2, y2]
    """
    # unwrap Tensors
    if hasattr(boxes1, "data"):
        boxes1 = boxes1.data
    if hasattr(boxes2, "data"):
        boxes2 = boxes2.data

    if boxes1.size == 0 or boxes2.size == 0:
        return xp.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=xp.float32)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # broadcast to (N, M)
    x1 = xp.maximum(x11[:, None], x21[None, :])
    y1 = xp.maximum(y11[:, None], y21[None, :])
    x2 = xp.minimum(x12[:, None], x22[None, :])
    y2 = xp.minimum(y12[:, None], y22[None, :])

    inter_w = xp.clip(x2 - x1, a_min=0, a_max=None)
    inter_h = xp.clip(y2 - y1, a_min=0, a_max=None)
    inter = inter_w * inter_h

    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)


def mean_average_precision(
    pred_boxes: List[Tensor],
    pred_scores: List[Tensor],
    pred_labels: List[Tensor],
    gt_boxes: List[Tensor],
    gt_labels: List[Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
    eps: float = 1e-12,
) -> Tuple[float, xp.ndarray]:
    """
    VOC-style mAP@IoU threshold.

    Args:
        pred_boxes:  list of (Pi, 4) Tensors - predicted boxes per image
        pred_scores: list of (Pi,) Tensors - scores per predicted box
        pred_labels: list of (Pi,) Tensors - predicted class ids
        gt_boxes:    list of (Gi, 4) Tensors - ground truth boxes per image
        gt_labels:   list of (Gi,) Tensors - GT class ids
        num_classes: number of classes (int)
        iou_threshold: IoU threshold to consider a prediction TP
        eps: small constant

    Returns:
        mAP: float - mean of per-class AP over classes with at least one GT
        per_class_ap: xp.ndarray of shape (num_classes,)
    """

    # Count GT boxes per class
    gt_per_class = xp.zeros(num_classes, dtype=xp.int64)
    for g_lbl in gt_labels:
        g = g_lbl.data
        for c in range(num_classes):
            gt_per_class[c] += xp.sum(g == c)

    per_class_ap = xp.zeros(num_classes, dtype=xp.float32)

    # Process each class independently (one-vs-rest)
    for c in range(num_classes):
        n_gt = int(gt_per_class[c])
        if n_gt == 0:
            # no GT for this class → ignore in mAP
            per_class_ap[c] = 0.0
            continue

        # Collect predictions of class c across all images
        class_preds = []  # list of (score, image_idx, box_tensor)
        for img_idx, (p_boxes, p_scores, p_labels) in enumerate(
            zip(pred_boxes, pred_scores, pred_labels)
        ):
            lb = p_labels.data
            mask = (lb == c)
            if xp.any(mask):
                b = p_boxes.data[mask]
                s = p_scores.data[mask]
                for j in range(b.shape[0]):
                    class_preds.append((float(s[j]), img_idx, b[j]))

        if len(class_preds) == 0:
            per_class_ap[c] = 0.0
            continue

        # Sort predictions by descending score
        class_preds.sort(key=lambda x: x[0], reverse=True)

        # Build map from image_idx -> GT boxes of class c
        gt_boxes_c = {}
        gt_used_c = {}

        for img_idx, (g_boxes, g_labels) in enumerate(zip(gt_boxes, gt_labels)):
            g_lbl = g_labels.data
            mask = (g_lbl == c)
            if xp.any(mask):
                boxes_c = g_boxes.data[mask]
                gt_boxes_c[img_idx] = boxes_c
                gt_used_c[img_idx] = xp.zeros(boxes_c.shape[0], dtype=bool)

        # Evaluate predictions as TP / FP
        tp = xp.zeros(len(class_preds), dtype=xp.float32)
        fp = xp.zeros(len(class_preds), dtype=xp.float32)

        for i, (_, img_idx, box_pred) in enumerate(class_preds):
            if img_idx not in gt_boxes_c:
                # no GT of this class in this image → FP
                fp[i] = 1.0
                continue

            g_boxes = gt_boxes_c[img_idx]
            used = gt_used_c[img_idx]

            # IoU between this pred and all GT boxes of class c in this image
            ious = box_iou(box_pred[None, :], g_boxes)[0]  # shape (G,)

            best_iou_idx = int(xp.argmax(ious))
            best_iou = float(ious[best_iou_idx])

            if best_iou >= iou_threshold and not bool(used[best_iou_idx]):
                tp[i] = 1.0
                used[best_iou_idx] = True
            else:
                fp[i] = 1.0

        # cumulative TP / FP
        tp_cum = xp.cumsum(tp)
        fp_cum = xp.cumsum(fp)

        recalls = tp_cum / (n_gt + eps)
        precisions = tp_cum / (tp_cum + fp_cum + eps)

        # AP: integrate precision–recall curve (VOC2010+ style)
        # add sentinel points
        mrec = xp.concatenate([xp.array([0.0]), recalls, xp.array([1.0])])
        mpre = xp.concatenate([xp.array([0.0]), precisions, xp.array([0.0])])

        # precision envelope
        for i in range(mpre.shape[0] - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # sum over recall changes
        idx = xp.where(mrec[1:] != mrec[:-1])[0]
        ap = xp.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        per_class_ap[c] = float(ap)

    # mAP over classes that actually appear in GT
    valid = gt_per_class > 0
    if xp.any(valid):
        mAP = float(xp.sum(per_class_ap[valid]) / (xp.sum(valid) + eps))
    else:
        mAP = 0.0

    return mAP, per_class_ap


def dice_coefficient(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-12,
):
    """
    Dice coefficient for segmentation.

    Supports:
      1) Multi-class with logits:
         preds:   (N, C, H, W)  - raw scores / logits
         targets: (N, H, W)     - integer class labels

      2) Binary / multi-label masks:
         preds:   (N, H, W)     or (N, C, H, W)
         targets: same shape as preds

    Returns:
        micro_dice: float
        macro_dice: float
        weighted_dice: float
        per_class_dice: xp.ndarray of shape (C,)
    """
    p = preds.data
    t = targets.data

    # ---------------- CASE 1: logits (N, C, H, W) + class indices (N, H, W) ----------------
    if p.ndim == 4 and t.ndim == 3:
        N, C, H, W = p.shape

        # predicted labels
        pred_labels = xp.argmax(p, axis=1)  # (N, H, W)
        true_labels = t                     # (N, H, W)

        # flatten
        pred_flat = pred_labels.reshape(-1)
        true_flat = true_labels.reshape(-1)

        C = int(max(pred_flat.max(), true_flat.max()) + 1)

        intersection = xp.zeros(C, dtype=xp.int64)
        pred_sum = xp.zeros(C, dtype=xp.int64)
        true_sum = xp.zeros(C, dtype=xp.int64)

        for c in range(C):
            pred_c = (pred_flat == c)
            true_c = (true_flat == c)

            intersection[c] = xp.sum(pred_c & true_c)
            pred_sum[c] = xp.sum(pred_c)
            true_sum[c] = xp.sum(true_c)

        per_class_dice = (2 * intersection) / (pred_sum + true_sum + eps)

        # micro Dice over all classes
        total_inter = xp.sum(intersection)
        total_pred = xp.sum(pred_sum)
        total_true = xp.sum(true_sum)
        micro_dice = float(2 * total_inter / (total_pred + total_true + eps))

        macro_dice = float(xp.mean(per_class_dice))

        # weighted by GT pixels per class
        support = true_sum
        total_support = xp.sum(support) + eps
        weighted_dice = float(xp.sum(per_class_dice * support) / total_support)

        return micro_dice, macro_dice, weighted_dice, per_class_dice

    # ---------------- CASE 2: masks with same shape ----------------
    if p.shape != t.shape:
        raise ValueError(
            "For mask Dice, preds and targets must have the same shape "
            "or be (N,C,H,W)/(N,H,W) for logits+labels."
        )

    # threshold predictions if not integer
    if p.dtype not in [xp.int32, xp.int64, xp.uint8]:
        preds_bin = (p > threshold).astype(xp.int32)
    else:
        preds_bin = p

    # binarize targets: any non-zero = 1
    if t.dtype not in [xp.int32, xp.int64, xp.uint8]:
        targets_bin = (t > 0.5).astype(xp.int32)
    else:
        targets_bin = (t != 0).astype(xp.int32)

    # (N, H, W) → single class
    if preds_bin.ndim == 3:
        preds_flat = preds_bin.reshape(1, -1)
        targets_flat = targets_bin.reshape(1, -1)
        C = 1

    # (N, C, H, W) → multi-label / multi-class one-hot
    elif preds_bin.ndim == 4:
        N, C, H, W = preds_bin.shape
        preds_flat = preds_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
        targets_flat = targets_bin.reshape(N, C, -1).transpose(1, 0, 2).reshape(C, -1)
    else:
        raise ValueError("Unsupported mask shape for Dice: expected (N,H,W) or (N,C,H,W).")

    intersection = xp.sum((preds_flat == 1) & (targets_flat == 1), axis=1)
    pred_sum = xp.sum(preds_flat == 1, axis=1)
    true_sum = xp.sum(targets_flat == 1, axis=1)

    per_class_dice = (2 * intersection) / (pred_sum + true_sum + eps)

    total_inter = xp.sum(intersection)
    total_pred = xp.sum(pred_sum)
    total_true = xp.sum(true_sum)

    micro_dice = float(2 * total_inter / (total_pred + total_true + eps))
    macro_dice = float(xp.mean(per_class_dice))

    support = true_sum
    total_support = xp.sum(support) + eps
    weighted_dice = float(xp.sum(per_class_dice * support) / total_support)

    return micro_dice, macro_dice, weighted_dice, per_class_dice


def _activation_stats(acts: Tensor, eps: float = 1e-6):
    """
    Compute mean and covariance of activations.

    acts: Tensor of shape (N, D)
    Returns:
        mu: xp.ndarray, shape (D,)
        cov: xp.ndarray, shape (D, D)
    """
    x = acts.data.astype(xp.float64)
    if x.ndim != 2:
        raise ValueError("FID expects activations of shape (N, D).")
    N, D = x.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to compute covariance.")

    mu = xp.mean(x, axis=0)

    xc = x - mu
    # unbiased covariance: (X^T X) / (N - 1)
    cov = (xc.T @ xc) / (N - 1 + eps)
    return mu, cov


def _matrix_sqrt(mat: xp.ndarray, eps: float = 1e-6) -> xp.ndarray:
    """
    Symmetric PSD matrix square root using eigen-decomposition.
    mat: (D, D), assumed symmetric.
    """
    # ensure symmetry (numerical)
    mat = 0.5 * (mat + mat.T)

    # eigen decomposition
    w, v = xp.linalg.eigh(mat)  # for symmetric
    w = xp.clip(w, a_min=0.0, a_max=None)
    sqrt_w = xp.sqrt(w + eps)

    return (v * sqrt_w[None, :]) @ v.T


def fid(
    real_acts: Tensor,
    fake_acts: Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Fréchet Inception Distance between real and generated activations.

    Args:
        real_acts: Tensor of shape (N_r, D)
        fake_acts: Tensor of shape (N_f, D)
        eps: small stabilizer

    Returns:
        Scalar float FID.
    """
    mu_r, cov_r = _activation_stats(real_acts, eps)
    mu_g, cov_g = _activation_stats(fake_acts, eps)

    # mean difference term
    diff = mu_r - mu_g
    diff_sq = float(diff @ diff)

    # covariance term
    cov_prod = cov_r @ cov_g
    cov_prod_sqrt = _matrix_sqrt(cov_prod + eps * xp.eye(cov_r.shape[0], dtype=cov_r.dtype), eps=eps)

    trace_r = float(xp.trace(cov_r))
    trace_g = float(xp.trace(cov_g))
    trace_sqrt = float(xp.trace(cov_prod_sqrt))

    fid_value = diff_sq + trace_r + trace_g - 2.0 * trace_sqrt

    # numerical noise can make this slightly negative
    if fid_value < 0:
        fid_value = 0.0

    return float(fid_value)


def inception_score(
    logits: Tensor,
    eps: float = 1e-12
) -> float:
    """
    Inception Score.

    Args:
        logits: Tensor of shape (N, C)
                model outputs BEFORE softmax for generated images.

    Returns:
        float IS
    """
    # compute p(y|x)
    # use your ops.log_softmax since it's stable
    log_probs = ops.log_softmax(logits, axis=1, epsilon=eps).data
    probs = xp.exp(log_probs)  # shape (N, C)

    # p(y) = mean over samples
    py = xp.mean(probs, axis=0)  # shape (C,)

    # KL divergence per sample:
    # sum_c  p(y|x) * (log p(y|x) - log p(y))
    log_py = xp.log(py + eps)
    kl_per_sample = xp.sum(probs * (log_probs - log_py[None, :]), axis=1)

    # average KL
    kl_mean = xp.mean(kl_per_sample)

    # IS = exp(expected KL)
    return float(xp.exp(kl_mean))


def inception_score_split(
    logits: Tensor,
    num_splits: int = 10,
    eps: float = 1e-12
) -> Tuple[float, float, xp.ndarray]:
    """
    Split Inception Score.

    Args:
        logits: Tensor of shape (N, C)
        num_splits: number of splits (default 10)
        eps: numeric stabilizer

    Returns:
        mean_IS: float
        std_IS: float
        split_IS: xp.ndarray of shape (num_splits,)
    """
    x = logits.data
    N = x.shape[0]

    if N < num_splits:
        raise ValueError("Number of samples must be >= num_splits.")

    # Compute log-softmax ONCE
    log_probs = ops.log_softmax(logits, axis=1, epsilon=eps).data
    probs = xp.exp(log_probs)

    # Split into chunks
    split_size = N // num_splits
    split_scores = []

    for k in range(num_splits):
        start = k * split_size
        end = (k + 1) * split_size

        p_yx = probs[start:end]          # (split_size, C)
        p_y = xp.mean(p_yx, axis=0)      # (C,)
        log_p_y = xp.log(p_y + eps)

        # KL divergence for the split
        kl = xp.sum(p_yx * (log_probs[start:end] - log_p_y[None, :]), axis=1)
        kl_mean = xp.mean(kl)

        split_scores.append(float(xp.exp(kl_mean)))

    split_scores = xp.array(split_scores, dtype=xp.float32)

    mean_IS = float(xp.mean(split_scores))
    std_IS = float(xp.std(split_scores))

    return mean_IS, std_IS, split_scores


def _gaussian_kernel(size: int, sigma: float, xp):
    ax = xp.arange(size) - size // 2
    kernel = xp.exp(-(ax**2) / (2 * sigma**2))
    kernel = kernel / xp.sum(kernel)
    return kernel


def _gaussian_filter(img, size: int, sigma: float, xp):
    # separable filter: conv along H then W
    kernel = _gaussian_kernel(size, sigma, xp)
    kernel = kernel.reshape(1, 1, 1, size)  # horizontal
    img = xp.apply_along_axis(lambda m: xp.convolve(m, kernel[0,0,0], mode="same"), 3, img)
    kernel = kernel.transpose(0,1,3,2)      # vertical
    img = xp.apply_along_axis(lambda m: xp.convolve(m, kernel[0,0,:,0], mode="same"), 2, img)
    return img


def ssim(
    img1: Tensor,
    img2: Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-12
) -> float:
    """
    True windowed SSIM using Gaussian kernel.
    img1, img2: (N, C, H, W) Tensors
    """
    x = img1.data.astype(xp.float64)
    y = img2.data.astype(xp.float64)

    # Gaussian blur for local statistics
    mu_x = _gaussian_filter(x, kernel_size, sigma, xp)
    mu_y = _gaussian_filter(y, kernel_size, sigma, xp)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _gaussian_filter(x * x, kernel_size, sigma, xp) - mu_x2
    sigma_y2 = _gaussian_filter(y * y, kernel_size, sigma, xp) - mu_y2
    sigma_xy = _gaussian_filter(x * y, kernel_size, sigma, xp) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)

    # mean over channels and spatial dims
    return float(xp.mean(ssim_map))