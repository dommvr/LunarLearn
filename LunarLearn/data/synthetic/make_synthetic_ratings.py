import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng

xp = backend.xp
DTYPE = backend.DTYPE


def make_synthetic_ratings(
    n_users=1000,
    n_items=1500,
    rank=20,
    sparsity=0.98,          # fraction missing (0.98 => 2% observed)
    noise_std=0.1,
    biased=True,
    rating_scale=(1.0, 5.0),
    implicit=False,

    # NEW: missingness model
    missingness="uniform",      # "uniform" or "popularity"
    pop_alpha=1.0,              # item popularity skew strength (higher => more head-heavy)
    act_alpha=0.5,              # user activity skew strength
    propensity_clip=0.95,       # clip observation probs to avoid p=1 everywhere

    seed_timestamp=False,
    random_state=None,
    dtype=None,
    return_factors=False,
):
    """
    Synthetic recommender ratings generator with optional popularity/activity-biased missingness.
    Returns triplets (user_idx, item_idx, rating_or_label [, timestamps]).
    """
    if dtype is None:
        dtype = DTYPE
    if not (0.0 <= sparsity < 1.0):
        raise ValueError("sparsity must be in [0, 1)")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if rating_scale[1] <= rating_scale[0]:
        raise ValueError("rating_scale must be (min, max) with max > min")
    if missingness not in ("uniform", "popularity"):
        raise ValueError('missingness must be "uniform" or "popularity"')
    if propensity_clip <= 0 or propensity_clip > 1.0:
        raise ValueError("propensity_clip must be in (0, 1]")

    rng = _get_rng(random_state)

    # Latent factors
    U = rng.normal(0.0, 1.0, size=(n_users, rank)).astype(dtype)
    V = rng.normal(0.0, 1.0, size=(n_items, rank)).astype(dtype)

    # Base scores
    S = U @ V.T  # (n_users, n_items)

    mu = xp.asarray(0.0, dtype=dtype)
    bu = xp.zeros((n_users,), dtype=dtype)
    bi = xp.zeros((n_items,), dtype=dtype)

    if biased:
        mu = xp.asarray(rng.normal(0.0, 0.5), dtype=dtype)
        bu = rng.normal(0.0, 0.5, size=(n_users,)).astype(dtype)
        bi = rng.normal(0.0, 0.5, size=(n_items,)).astype(dtype)
        S = S + mu + bu[:, None] + bi[None, :]

    if noise_std and noise_std > 0:
        S = S + rng.normal(0.0, noise_std, size=S.shape).astype(dtype)

    # Map/clip to rating scale
    rmin, rmax = float(rating_scale[0]), float(rating_scale[1])
    S = xp.tanh(S)  # [-1, 1]
    R = ((S + 1.0) * 0.5) * (rmax - rmin) + rmin
    R = xp.clip(R, rmin, rmax).astype(dtype)

    # --- NEW: missingness (uniform vs popularity/activity-biased) ---
    p_obs = 1.0 - float(sparsity)

    if missingness == "uniform":
        mask = rng.uniform(0.0, 1.0, size=(n_users, n_items)) < p_obs

    else:
        # Item popularity propensities (head-heavy via Zipf-like shape)
        # Using ranks avoids giant exponentials and works in both numpy/cupy
        item_rank = xp.arange(1, n_items + 1, dtype=dtype)  # 1..n_items
        item_pop = (1.0 / item_rank) ** xp.asarray(pop_alpha, dtype=dtype)
        item_pop = item_pop / (xp.mean(item_pop) + xp.asarray(1e-12, dtype=dtype))  # mean ~ 1

        # User activity propensities (some users rate more)
        user_rank = xp.arange(1, n_users + 1, dtype=dtype)
        user_act = (1.0 / user_rank) ** xp.asarray(act_alpha, dtype=dtype)
        user_act = user_act / (xp.mean(user_act) + xp.asarray(1e-12, dtype=dtype))  # mean ~ 1

        # Outer product gives relative propensities per (u, i)
        base = user_act[:, None] * item_pop[None, :]  # mean ~ 1

        # Scale so mean(prob) == p_obs, then clip
        probs = xp.asarray(p_obs, dtype=dtype) * base
        probs = xp.clip(probs, 0.0, float(propensity_clip)).astype(dtype)

        mask = rng.uniform(0.0, 1.0, size=(n_users, n_items)) < probs

    user_idx, item_idx = xp.where(mask)
    ratings = R[user_idx, item_idx]

    # Optional timestamps
    timestamps = None
    if seed_timestamp:
        n_obs = int(user_idx.shape[0])
        t = rng.permutation(n_obs).astype(xp.int64)
        jitter = rng.randint(0, 10, size=(n_obs,), dtype=xp.int64)
        timestamps = (t * 10 + jitter).astype(xp.int64)

    if implicit:
        thresh = (rmin + rmax) * 0.5
        y = (ratings >= xp.asarray(thresh, dtype=dtype)).astype(xp.int64)
        if return_factors:
            return (user_idx, item_idx, y, timestamps, U, V, mu, bu, bi) if seed_timestamp else (user_idx, item_idx, y, U, V, mu, bu, bi)
        return (user_idx, item_idx, y, timestamps) if seed_timestamp else (user_idx, item_idx, y)

    if return_factors:
        return (user_idx, item_idx, ratings, timestamps, U, V, mu, bu, bi) if seed_timestamp else (user_idx, item_idx, ratings, U, V, mu, bu, bi)
    return (user_idx, item_idx, ratings, timestamps) if seed_timestamp else (user_idx, item_idx, ratings)