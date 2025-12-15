import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _shift_for_next_token,
                                             _pad_1d)

xp = backend.xp
DTYPE = backend.DTYPE


def make_addition_next_token(
    n_samples=2000,
    max_digits=3,
    pad_token=0,
    bos_token=1,
    eos_token=2,
    plus_token=13,
    eq_token=14,
    random_state=None,
):
    """
    Next-token prediction for full expression "a+b=c".
      seq: [BOS, digits(a), '+', digits(b), '=', digits(c), EOS] padded to T
      X = seq[:, :-1], Y = seq[:, 1:]

    Returns:
      seq: (N, T) int64
      X: (N, T-1)
      Y: (N, T-1)
      lengths: (N,) length incl BOS/EOS
    """
    rng = _get_rng(random_state)
    D = int(max_digits)
    if D < 1:
        raise ValueError("max_digits must be >= 1")

    # Worst-case: a,b have D digits, sum has D+1 digits
    # BOS + D + 1 + D + 1 + (D+1) + EOS = 3D + 5
    T = 3 * D + 5

    seqs = xp.full((n_samples, T), pad_token, dtype=xp.int64)
    lengths = xp.empty((n_samples,), dtype=xp.int64)

    def enc_num(n):
        s = str(int(n))
        return [3 + int(ch) for ch in s]

    max_n = 10 ** D - 1

    for i in range(n_samples):
        a = int(rng.randint(0, max_n + 1))
        b = int(rng.randint(0, max_n + 1))
        c = a + b

        seq = [bos_token] + enc_num(a) + [plus_token] + enc_num(b) + [eq_token] + enc_num(c) + [eos_token]
        padded, n = _pad_1d(seq, T, pad_token)

        seqs[i] = xp.asarray(padded, dtype=xp.int64)
        lengths[i] = n

    X, Y = _shift_for_next_token(seqs, pad_token)
    return seqs, X, Y, lengths