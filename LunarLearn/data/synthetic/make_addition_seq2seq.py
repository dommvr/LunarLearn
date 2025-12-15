import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _pad_1d)

xp = backend.xp
DTYPE = backend.DTYPE


def make_addition_seq2seq(
    n_samples=2000,
    max_digits=3,
    pad_token=0,
    bos_token=1,
    eos_token=2,
    plus_token=13,
    eq_token=14,
    # digits 0..9 map to tokens 3..12
    random_state=None,
):
    """
    Seq2seq addition:
      src: [BOS, digits(a), '+', digits(b), '=', EOS]  (padded)
      tgt: [BOS, digits(a+b), EOS]                     (padded)

    Tokenization:
      digit d -> token (3 + d)
      plus -> plus_token, eq -> eq_token
    Returns:
      src: (N, Ts) int64
      tgt: (N, Tt) int64
      src_len, tgt_len: (N,) int64
    """
    rng = _get_rng(random_state)

    if max_digits < 1:
        raise ValueError("max_digits must be >= 1")

    # Max lengths:
    # src: BOS + max_digits + 1 + max_digits + 1 + EOS = 2*D + 4
    # tgt: BOS + (max_digits+1) + EOS = D + 3
    D = int(max_digits)
    Ts = 2 * D + 4
    Tt = D + 3

    src = xp.full((n_samples, Ts), pad_token, dtype=xp.int64)
    tgt = xp.full((n_samples, Tt), pad_token, dtype=xp.int64)
    src_len = xp.empty((n_samples,), dtype=xp.int64)
    tgt_len = xp.empty((n_samples,), dtype=xp.int64)

    def enc_num(n):
        # no leading zeros (unless n == 0)
        s = str(int(n))
        return [3 + int(ch) for ch in s]

    max_n = 10 ** D - 1

    for i in range(n_samples):
        a = int(rng.randint(0, max_n + 1))
        b = int(rng.randint(0, max_n + 1))
        c = a + b

        src_seq = [bos_token] + enc_num(a) + [plus_token] + enc_num(b) + [eq_token] + [eos_token]
        tgt_seq = [bos_token] + enc_num(c) + [eos_token]

        psrc, ns = _pad_1d(src_seq, Ts, pad_token)
        ptgt, nt = _pad_1d(tgt_seq, Tt, pad_token)

        src[i] = xp.asarray(psrc, dtype=xp.int64)
        tgt[i] = xp.asarray(ptgt, dtype=xp.int64)
        src_len[i] = ns
        tgt_len[i] = nt

    return src, tgt, src_len, tgt_len