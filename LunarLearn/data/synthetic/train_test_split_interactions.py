import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng

xp = backend.xp


def train_test_split_interactions(
    user_idx,
    item_idx,
    values,
    test_size=0.2,
    shuffle=True,
    random_state=None,
    timestamps=None,
):
    """
    Split interaction triplets either randomly or chronologically (if timestamps provided).

    Inputs:
        user_idx, item_idx: 1D arrays
        values: ratings or implicit labels, 1D
        timestamps: optional 1D, same length. If provided, split by time (latest -> test).

    Returns:
        (u_tr, i_tr, v_tr), (u_te, i_te, v_te)
    """
    rng = _get_rng(random_state)

    n = int(user_idx.shape[0])
    if n == 0:
        raise ValueError("No interactions to split")

    n_test = int(round(float(test_size) * n))
    n_test = max(1, min(n_test, n - 1))

    if timestamps is not None:
        order = xp.argsort(timestamps)  # oldest -> newest
    else:
        order = rng.permutation(n) if shuffle else xp.arange(n, dtype=xp.int64)

    te_idx = order[-n_test:]
    tr_idx = order[:-n_test]

    u_tr, i_tr, v_tr = user_idx[tr_idx], item_idx[tr_idx], values[tr_idx]
    u_te, i_te, v_te = user_idx[te_idx], item_idx[te_idx], values[te_idx]

    return (u_tr, i_tr, v_tr), (u_te, i_te, v_te)