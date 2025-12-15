import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _make_grid

xp = backend.xp
DTYPE = backend.DTYPE


def make_bars_stripes(
    n_samples=5000,
    image_size=32,
    channels=1,                  # 1 or 3
    orientations=("h", "v", "d"), # horizontal, vertical, diagonal (main)
    freq_range=(2, 10),          # number of stripe periods across the image
    thickness_range=(0.25, 0.75),# duty cycle: fraction of a period that is "on"
    phase_jitter=True,           # random phase shift
    invert_prob=0.5,             # probability to invert colors (swap fg/bg)
    contrast_range=(0.7, 1.0),   # stripe intensity
    bg_range=(0.0, 0.2),         # background intensity
    noise_std=0.05,
    blur_strength=0.0,           # cheap 3x3 blur if >0 (0..1)
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Bars & Stripes dataset (NCHW).
    Labels correspond to orientation index in `orientations`.

    Returns:
        X: (n_samples, C, H, W) float in [0,1]
        y: (n_samples,) int64
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if H < 8:
        raise ValueError("image_size too small")
    if len(orientations) < 2:
        raise ValueError("provide at least 2 orientations")
    if freq_range[0] < 1 or freq_range[1] < freq_range[0]:
        raise ValueError("invalid freq_range")
    if not (0.0 < thickness_range[0] <= thickness_range[1] < 1.0):
        raise ValueError("thickness_range must be in (0,1) with lo<=hi")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    n_classes = len(orientations)
    y = rng.randint(0, n_classes, size=(n_samples,), dtype=xp.int64)

    X = xp.zeros((n_samples, channels, H, W), dtype=dtype)

    # Normalized coordinates in [0,1)
    # Use eps to avoid boundary quirks
    eps = xp.finfo(dtype).eps
    yn = yy / xp.asarray(max(H - 1, 1), dtype=dtype)
    xn = xx / xp.asarray(max(W - 1, 1), dtype=dtype)
    yn = xp.clip(yn, 0.0, 1.0 - eps)
    xn = xp.clip(xn, 0.0, 1.0 - eps)

    # Optional cheap blur kernel (separable-ish 3x3 box)
    def _blur2d(img2d, strength):
        if strength <= 0:
            return img2d
        # 3x3 box blur implemented with padding + sums (no fancy conv)
        pad = 1
        H0, W0 = img2d.shape
        tmp = xp.pad(img2d, ((pad, pad), (pad, pad)), mode="edge")
        s = (
            tmp[0:H0,     0:W0]     + tmp[0:H0,     1:W0+1]     + tmp[0:H0,     2:W0+2] +
            tmp[1:H0+1,   0:W0]     + tmp[1:H0+1,   1:W0+1]     + tmp[1:H0+1,   2:W0+2] +
            tmp[2:H0+2,   0:W0]     + tmp[2:H0+2,   1:W0+1]     + tmp[2:H0+2,   2:W0+2]
        ) / xp.asarray(9.0, dtype=dtype)
        return (1.0 - strength) * img2d + strength * s

    for i in range(n_samples):
        ori = orientations[int(y[i])]

        # Choose frequency (period count) and duty cycle
        freq = int(rng.randint(int(freq_range[0]), int(freq_range[1]) + 1))
        duty = float(rng.uniform(float(thickness_range[0]), float(thickness_range[1])))

        # Phase
        phase = float(rng.uniform(0.0, 1.0)) if phase_jitter else 0.0

        # Background + foreground intensity
        bg = float(rng.uniform(float(bg_range[0]), float(bg_range[1])))
        fg = float(rng.uniform(float(contrast_range[0]), float(contrast_range[1])))

        # Build coordinate "t" for stripes
        # t in [0,1): stripes defined by frac(freq * t + phase) < duty
        if ori == "h":
            t = yn
        elif ori == "v":
            t = xn
        elif ori == "d":
            # main diagonal stripes (top-left to bottom-right)
            t = (xn + yn) * 0.5  # still in ~[0,1]
        elif ori == "ad":
            # anti-diagonal stripes (top-right to bottom-left)
            t = (xn - yn) * 0.5 + 0.5
        else:
            raise ValueError(f"Unknown orientation '{ori}' (use 'h','v','d','ad')")

        frac = (xp.asarray(freq, dtype=dtype) * t + xp.asarray(phase, dtype=dtype)) % xp.asarray(1.0, dtype=dtype)
        mask = frac < xp.asarray(duty, dtype=dtype)

        img2d = xp.full((H, W), bg, dtype=dtype)
        img2d[mask] = xp.asarray(fg, dtype=dtype)

        # Invert colors sometimes
        if float(rng.uniform(0.0, 1.0)) < float(invert_prob):
            img2d = (xp.asarray(1.0, dtype=dtype) - img2d)

        # Blur (optional)
        if blur_strength and blur_strength > 0:
            img2d = _blur2d(img2d, float(blur_strength))

        # Noise
        if noise_std and noise_std > 0:
            img2d = img2d + rng.normal(0.0, float(noise_std), size=img2d.shape).astype(dtype)

        img2d = xp.clip(img2d, 0.0, 1.0)

        # Write to NCHW
        if channels == 1:
            X[i, 0] = img2d
        else:
            # slight per-channel variation so it's not trivially grayscale unless you want that
            jitter = rng.uniform(0.0, 0.08, size=(3,)).astype(dtype)
            for c in range(3):
                X[i, c] = xp.clip(img2d + jitter[c], 0.0, 1.0)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y