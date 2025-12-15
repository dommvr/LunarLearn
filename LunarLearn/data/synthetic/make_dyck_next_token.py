import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _shift_for_next_token,
                                             _pad_1d)

xp = backend.xp
DTYPE = backend.DTYPE


def make_dyck_next_token(
    n_samples=2000,
    max_len=64,        # total length including BOS, EOS (fixed length output)
    max_depth=8,
    pad_token=0,
    bos_token=1,
    eos_token=2,
    open_token=3,      # "("
    close_token=4,     # ")"
    random_state=None,
):
    """
    Generates balanced parentheses strings (Dyck-1), then returns next-token prediction pairs.

    Output:
      seq: (N, T) int64 = [BOS, ...parens..., EOS, PAD...]
      X, Y: next-token inputs/targets where:
        X = seq[:, :-1]
        Y = seq[:, 1:]
      lengths: (N,) true length including BOS/EOS
    """
    rng = _get_rng(random_state)
    T = int(max_len)

    seqs = xp.full((n_samples, T), pad_token, dtype=xp.int64)
    lengths = xp.empty((n_samples,), dtype=xp.int64)

    for i in range(n_samples):
        # Build a balanced sequence by random stack actions with constraints
        stack = 0
        tokens = [bos_token]

        # We generate until we decide to stop, then close all opens, then add EOS.
        # Keep it under T.
        while True:
            # Remaining room must fit: closes for current stack + EOS
            remaining = T - len(tokens)
            if remaining <= (stack + 1):
                break

            # Stop with some probability once we have something
            if len(tokens) > 2 and float(rng.uniform(0.0, 1.0)) < 0.05:
                break

            # Choose open vs close
            if stack == 0:
                action = "open"
            elif stack >= int(max_depth):
                action = "close"
            else:
                action = "open" if float(rng.uniform(0.0, 1.0)) < 0.5 else "close"

            if action == "open":
                tokens.append(open_token)
                stack += 1
            else:
                tokens.append(close_token)
                stack -= 1

        # Close everything
        while stack > 0 and (len(tokens) + 1) < T:
            tokens.append(close_token)
            stack -= 1

        # EOS
        if len(tokens) < T:
            tokens.append(eos_token)

        padded, n = _pad_1d(tokens, T, pad_token)
        seqs[i] = xp.asarray(padded, dtype=xp.int64)
        lengths[i] = n

    X, Y = _shift_for_next_token(seqs, pad_token)
    return seqs, X, Y, lengths
