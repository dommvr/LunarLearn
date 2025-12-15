import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _render_perlin_like_blobs,
                                             _render_checkerboard,
                                             _render_stripes,
                                             _render_gradient,
                                             _normalize01,
                                             _to_nchw)

xp = backend.xp
DTYPE = backend.DTYPE


def make_procedural_textures(
    n_samples=2000,
    image_size=64,
    channels=1,  # NCHW
    modes=("blobs", "checkerboard", "stripes", "gradient"),
    mode_weights=None,          # list same length as modes
    mix_prob=0.25,              # sometimes mix two textures
    noise_std=0.02,
    contrast_jitter=(0.8, 1.2),
    brightness_jitter=(-0.1, 0.1),
    shuffle=True,
    random_state=None,
    dtype=None,
    return_labels=False,        # if True, returns y = texture type index
):
    """
    Procedural texture generator for unconditional generative modeling.
    Returns:
        X: (N,C,H,W) in [0,1]
        (optional) y: (N,) int64, index into modes
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if len(modes) < 1:
        raise ValueError("modes must be non-empty")

    rng = _get_rng(random_state)

    # weights
    if mode_weights is None:
        probs = xp.ones((len(modes),), dtype=dtype) / float(len(modes))
    else:
        if len(mode_weights) != len(modes):
            raise ValueError("mode_weights must match modes length")
        s = float(sum(mode_weights))
        if s <= 0:
            raise ValueError("mode_weights must sum to > 0")
        probs = xp.asarray([w / s for w in mode_weights], dtype=dtype)

    X = xp.zeros((n_samples, channels, H, W), dtype=dtype)
    y = xp.zeros((n_samples,), dtype=xp.int64)

    # helper to draw one texture
    def render_one(mode):
        if mode == "blobs":
            return _render_perlin_like_blobs(
                rng, H, W, dtype,
                octaves=int(rng.randint(3, 6)),
                base_block=max(2, H // int(rng.randint(10, 22))),
                blur_iters=int(rng.randint(1, 3)),
            )
        if mode == "checkerboard":
            return _render_checkerboard(rng, H, W, dtype, n_squares=int(rng.randint(2, 12)))
        if mode == "stripes":
            return _render_stripes(rng, H, W, dtype)
        if mode == "gradient":
            return _render_gradient(rng, H, W, dtype)
        raise ValueError(f"Unknown mode '{mode}'")

    for i in range(n_samples):
        # sample mode index
        # (xp.random.choice with p sometimes differs across backends; do it manually)
        r = float(rng.uniform(0.0, 1.0))
        acc = 0.0
        mi = 0
        for k in range(len(modes)):
            acc += float(probs[k])
            if r <= acc:
                mi = k
                break
        y[i] = mi

        tex = render_one(modes[mi])

        # Sometimes mix a second texture
        if float(rng.uniform(0.0, 1.0)) < float(mix_prob) and len(modes) > 1:
            j = int(rng.randint(0, len(modes)))
            tex2 = render_one(modes[j])
            alpha = float(rng.uniform(0.25, 0.75))
            tex = xp.asarray(alpha, dtype=dtype) * tex + xp.asarray(1.0 - alpha, dtype=dtype) * tex2
            tex = _normalize01(tex, dtype)

        # photometric jitter
        c = float(rng.uniform(float(contrast_jitter[0]), float(contrast_jitter[1])))
        b = float(rng.uniform(float(brightness_jitter[0]), float(brightness_jitter[1])))
        tex = xp.clip(xp.asarray(c, dtype=dtype) * tex + xp.asarray(b, dtype=dtype), 0.0, 1.0)

        img = _to_nchw(tex, channels, rng, dtype, colorize=True)

        # pixel noise
        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

        X[i] = xp.clip(img, 0.0, 1.0)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return (X.astype(dtype), y) if return_labels else X.astype(dtype)