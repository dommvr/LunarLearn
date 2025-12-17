import LunarLearn.core.backend.backend as backend
from LunarLearn.data.augmentation.utils import (_resolve_synonyms,
                                                simple_word_tokenize,
                                                simple_word_detokenize,
                                                _is_word,
                                                _permutation,
                                                _choice,
                                                _rand,
                                                _randint,
                                                _typo_char,
                                                _normalize_whitespace)
import string

xp = backend.xp


# ----------------------------
# 1) Classic EDA-style (text-level)
# ----------------------------

class SynonymReplacement:
    """
    Replace up to n words with synonyms.
    """
    def __init__(self, n=1, synonyms="small", tokenizer=simple_word_tokenize, detokenizer=simple_word_detokenize):
        self.n = int(n)
        self.syn = _resolve_synonyms(synonyms)
        self.tok = tokenizer
        self.detok = detokenizer

    def __call__(self, text):
        tokens = self.tok(text)
        word_pos = [i for i, t in enumerate(tokens) if _is_word(t)]
        if not word_pos:
            return text

        perm = _permutation(len(word_pos))
        replaced = 0

        for pi in perm:
            i = word_pos[pi]
            w = tokens[i]
            key = w.lower()
            cands = self.syn.get(key)
            if not cands:
                continue

            rep = _choice(cands)

            # preserve casing (best-effort)
            if w.isupper():
                rep = rep.upper()
            elif w[:1].isupper():
                rep = rep[:1].upper() + rep[1:]

            tokens[i] = rep
            replaced += 1
            if replaced >= self.n:
                break

        return self.detok(tokens)


class RandomDeletion:
    """
    Delete each word with probability p. Keeps punctuation tokens.
    """
    def __init__(self, p=0.1, tokenizer=simple_word_tokenize, detokenizer=simple_word_detokenize):
        self.p = float(p)
        self.tok = tokenizer
        self.detok = detokenizer

    def __call__(self, text):
        tokens = self.tok(text)
        if not tokens:
            return text

        out = []
        for t in tokens:
            if _is_word(t) and _rand() < self.p:
                continue
            out.append(t)

        if not out:
            out = [tokens[_randint(0, len(tokens))]]

        return self.detok(out)


class RandomSwap:
    """
    Swap two random word tokens n times.
    """
    def __init__(self, n=1, tokenizer=simple_word_tokenize, detokenizer=simple_word_detokenize):
        self.n = int(n)
        self.tok = tokenizer
        self.detok = detokenizer

    def __call__(self, text):
        tokens = self.tok(text)
        idx = [i for i, t in enumerate(tokens) if _is_word(t)]
        if len(idx) < 2:
            return text

        for _ in range(self.n):
            a = idx[_randint(0, len(idx))]
            b = idx[_randint(0, len(idx))]
            if a != b:
                tokens[a], tokens[b] = tokens[b], tokens[a]

        return self.detok(tokens)


class RandomInsertion:
    """
    Insert synonyms of random words n times.
    """
    def __init__(self, n=1, synonyms="small", tokenizer=simple_word_tokenize, detokenizer=simple_word_detokenize):
        self.n = int(n)
        self.syn = _resolve_synonyms(synonyms)
        self.tok = tokenizer
        self.detok = detokenizer

    def __call__(self, text):
        tokens = self.tok(text)
        word_pos = [i for i, t in enumerate(tokens) if _is_word(t)]
        if not word_pos:
            return text

        for _ in range(self.n):
            i = word_pos[_randint(0, len(word_pos))]
            w = tokens[i].lower()
            cands = self.syn.get(w)
            if not cands:
                continue
            ins = _choice(cands)
            pos = _randint(0, len(tokens) + 1)
            tokens.insert(pos, ins)

        return self.detok(tokens)


# ----------------------------
# 2) Human-ish noise (text-level)
# ----------------------------

class CharTypos:
    """
    Keyboard-neighbor typos per character with probability p_char.
    """
    def __init__(self, p_char=0.02):
        self.p = float(p_char)

    def __call__(self, text):
        out = []
        for c in text:
            if c.isalpha() and _rand() < self.p:
                out.append(_typo_char(c))
            else:
                out.append(c)
        return "".join(out)


class RandomCharDeleteInsert:
    """
    Randomly delete/insert characters.
    """
    def __init__(self, p_delete=0.01, p_insert=0.01, alphabet=None):
        self.pd = float(p_delete)
        self.pi = float(p_insert)
        self.alpha = alphabet if alphabet is not None else (string.ascii_lowercase + string.digits)

    def __call__(self, text):
        out = []
        for c in text:
            if _rand() < self.pd:
                continue
            out.append(c)
            if _rand() < self.pi:
                out.append(self.alpha[_randint(0, len(self.alpha))])
        return "".join(out)


class RandomCasing:
    def __init__(self, p=0.03):
        self.p = float(p)

    def __call__(self, text):
        out = []
        for c in text:
            if c.isalpha() and _rand() < self.p:
                out.append(c.swapcase())
            else:
                out.append(c)
        return "".join(out)


class PunctWhitespaceNoise:
    def __init__(self, p_space=0.05, p_punct=0.03):
        self.ps = float(p_space)
        self.pp = float(p_punct)
        self.punct = ["!", "?", ".", ",", ";", ":"]

    def __call__(self, text):
        chars = list(text)
        out = []
        for c in chars:
            if c.isspace() and _rand() < self.ps:
                # drop or duplicate whitespace
                if _rand() < 0.5:
                    continue
                out.append(c)
                out.append(c)
                continue

            out.append(c)

            if c.isalpha() and _rand() < self.pp:
                out.append(self.punct[_randint(0, len(self.punct))])

        return _normalize_whitespace("".join(out))


# ----------------------------
# 3) Model-ish augmentations (optional hooks)
# ----------------------------

class BackTranslation:
    """
    Optional. Provide translate_fn(text)->text. Otherwise no-op (unless strict=True).
    """
    def __init__(self, translate_fn=None, strict=False):
        self.fn = translate_fn
        self.strict = bool(strict)

    def __call__(self, text):
        if self.fn is None:
            if self.strict:
                raise RuntimeError("BackTranslation requires translate_fn")
            return text
        return self.fn(text)


class Paraphrase:
    """
    Optional. Provide paraphrase_fn(text)->text. Otherwise no-op (unless strict=True).
    """
    def __init__(self, paraphrase_fn=None, strict=False):
        self.fn = paraphrase_fn
        self.strict = bool(strict)

    def __call__(self, text):
        if self.fn is None:
            if self.strict:
                raise RuntimeError("Paraphrase requires paraphrase_fn")
            return text
        return self.fn(text)


# ----------------------------
# 4) Regularization-style
# ----------------------------

class WordDropout:
    """
    Drop words (text-level). Equivalent to RandomDeletion but semantically "regularization".
    """
    def __init__(self, p=0.1, tokenizer=simple_word_tokenize, detokenizer=simple_word_detokenize):
        self.p = float(p)
        self.tok = tokenizer
        self.detok = detokenizer

    def __call__(self, text):
        tokens = self.tok(text)
        if not tokens:
            return text
        out = []
        for t in tokens:
            if _is_word(t) and _rand() < self.p:
                continue
            out.append(t)
        if not out:
            out = [tokens[_randint(0, len(tokens))]]
        return self.detok(out)


class TokenDropoutIds:
    """
    Token dropout on integer token IDs: replace tokens with mask_id with prob p.
    Use this for transformer-ish regularization (works with your tokenizer output).

    ids: (T,) xp array of int64
    """
    def __init__(self, p=0.1, mask_id=0, keep_ids=None):
        self.p = float(p)
        self.mask_id = int(mask_id)
        self.keep_ids = set(int(x) for x in keep_ids) if keep_ids is not None else None

    def __call__(self, ids):
        x = xp.asarray(ids, dtype=xp.int64).copy()
        T = int(x.shape[0])
        if T == 0:
            return x
        r = xp.random.rand(T)
        drop = r < self.p
        if self.keep_ids is not None and len(self.keep_ids) > 0:
            keep_mask = xp.zeros((T,), dtype=xp.bool_)
            for kid in self.keep_ids:
                keep_mask = keep_mask | (x == kid)
            drop = drop & (~keep_mask)
        x[drop] = self.mask_id
        return x


class SpanMaskingIds:
    """
    Mask contiguous spans in token IDs.

    ids: (T,) int64
    """
    def __init__(self, p=0.15, span_len=(2, 5), mask_id=0, keep_ids=None):
        self.p = float(p)
        self.s0, self.s1 = int(span_len[0]), int(span_len[1])
        self.mask_id = int(mask_id)
        self.keep_ids = set(int(x) for x in keep_ids) if keep_ids is not None else None

    def __call__(self, ids):
        x = xp.asarray(ids, dtype=xp.int64).copy()
        T = int(x.shape[0])
        if T == 0 or self.p <= 0:
            return x

        # positions eligible for masking
        eligible = xp.ones((T,), dtype=xp.bool_)
        if self.keep_ids is not None and len(self.keep_ids) > 0:
            for kid in self.keep_ids:
                eligible = eligible & (x != kid)

        elig_idx = xp.where(eligible)[0]
        n_elig = int(elig_idx.shape[0])
        if n_elig == 0:
            return x

        # target roughly p*T eligible tokens masked
        target = max(1, int(round(self.p * n_elig)))
        masked = 0

        # attempt spans until target reached (bounded attempts)
        attempts = 0
        max_attempts = 4 * target + 16

        while masked < target and attempts < max_attempts:
            attempts += 1
            start = int(elig_idx[_randint(0, n_elig)])
            span = _randint(self.s0, self.s1 + 1)
            end = min(T, start + span)

            for i in range(start, end):
                if eligible[i] and x[i] != self.mask_id:
                    x[i] = self.mask_id
                    masked += 1
                    if masked >= target:
                        break

        return x


# ----------------------------
# Optional: token-level EDA using your tokenizer (encode/decode)
# (Useful if you want word ops but only have token IDs in the sample)
# ----------------------------

class TextAugmentViaTokenizer:
    """
    Apply a TEXT-level augmentation to token IDs using your tokenizer:
      ids -> decode -> augment(text) -> encode -> ids

    Warning: can change length and may break alignment tasks; use carefully.
    """
    def __init__(self, tokenizer, text_transform, encode_kwargs=None, decode_kwargs=None):
        self.tok = tokenizer
        self.tt = text_transform
        self.ek = encode_kwargs or {}
        self.dk = decode_kwargs or {}

    def __call__(self, ids):
        ids = xp.asarray(ids, dtype=xp.int64)
        # tokenizer likely expects python list/int array; keep it flexible
        text = self.tok.decode(ids.tolist(), **self.dk)
        text2 = self.tt(text)
        ids2 = self.tok.encode(text2, **self.ek)
        return xp.asarray(ids2, dtype=xp.int64)