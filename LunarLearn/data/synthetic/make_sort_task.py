import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _make_lengths,
                                             _pad_1d)

xp = backend.xp
DTYPE = backend.DTYPE


def make_sort_task(
    n_samples=2000,
    vocab_size=50,
    min_len=4,
    max_len=20,
    pad_token=0,
    bos_token=1,
    eos_token=2,
    random_state=None,
    key_mod=None,   # if not None: sort by (token % key_mod), stable
):
    """
    Sort task:
      src: [BOS, x1..xL, EOS]
      tgt: [BOS, sorted(x), EOS]

    If key_mod is set, sort by key = token % key_mod (stable),
    which makes "sort by a key" without needing pairs.
    """
    if vocab_size <= 3:
        raise ValueError("vocab_size must be > 3")
    rng = _get_rng(random_state)

    Ls = _make_lengths(rng, n_samples, min_len, max_len)
    T = int(max_len) + 2

    src = xp.full((n_samples, T), pad_token, dtype=xp.int64)
    tgt = xp.full((n_samples, T), pad_token, dtype=xp.int64)
    lengths = xp.empty((n_samples,), dtype=xp.int64)

    for i in range(n_samples):
        L = int(Ls[i])
        tokens = rng.randint(3, vocab_size, size=(L,), dtype=xp.int64).tolist()

        if key_mod is None:
            sorted_tokens = sorted(tokens)
        else:
            km = int(key_mod)
            sorted_tokens = sorted(tokens, key=lambda t: (t % km))

        src_seq = [bos_token] + tokens + [eos_token]
        tgt_seq = [bos_token] + sorted_tokens + [eos_token]

        psrc, n = _pad_1d(src_seq, T, pad_token)
        ptgt, _ = _pad_1d(tgt_seq, T, pad_token)
        src[i] = xp.asarray(psrc, dtype=xp.int64)
        tgt[i] = xp.asarray(ptgt, dtype=xp.int64)
        lengths[i] = n

    return src, tgt, lengths