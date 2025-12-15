import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng, _make_grid, _lowfreq_texture, _apply_color, _circle_mask, _rotated_rect_mask, _triangle_mask, _add_watermark)
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_shapes_shortcut(
    n_samples=4000,
    image_size=32,
    channels=1,                 # NCHW output
    shape_scale=(0.25, 0.55),
    rotation=True,
    translate=0.25,
    background="lowfreq",       # "none" | "uniform" | "lowfreq"
    bg_strength=(0.0, 0.35),
    clutter=2,                  # distractors (not labeled)
    clutter_scale=(0.08, 0.20),
    noise_std=0.05,

    # shortcut controls
    p_shortcut=0.90,            # watermark matches class with this probability
    watermark_size=6,           # pixels
    watermark_intensity=(0.6, 1.0),
    watermark_corners=("tl", "tr", "bl"),  # one corner per class (3 classes)

    flip_y=0.0,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Shortcut dataset: shapes classification + spurious corner watermark correlated with class.

    Labels:
      0 circle, 1 square, 2 triangle
    Returns:
      X: (N, C, H, W) in [0,1]
      y: (N,) int64
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if len(watermark_corners) != 3:
        raise ValueError("watermark_corners must have length 3 for 3 classes")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    y = rng.randint(0, 3, size=(n_samples,), dtype=xp.int64)

    # optional label noise
    if flip_y and flip_y > 0:
        n_flip = int(round(float(flip_y) * n_samples))
        if n_flip > 0:
            idx = rng.choice(n_samples, size=(n_flip,), replace=False)
            new_y = rng.randint(0, 3, size=(n_flip,), dtype=xp.int64)
            same = new_y == y[idx]
            if xp.any(same):
                new_y[same] = (new_y[same] + 1) % 3
            y[idx] = new_y

    X = xp.zeros((n_samples, channels, H, W), dtype=dtype)

    min_dim = float(min(H, W))
    s_lo = float(shape_scale[0]) * min_dim
    s_hi = float(shape_scale[1]) * min_dim
    c_lo = float(clutter_scale[0]) * min_dim
    c_hi = float(clutter_scale[1]) * min_dim

    base_cy = (H - 1) / 2.0
    base_cx = (W - 1) / 2.0
    max_shift = float(translate) * min_dim

    for i in range(n_samples):
        img = xp.zeros((channels, H, W), dtype=dtype)

        # Background
        if background == "uniform":
            bg = float(rng.uniform(bg_strength[0], bg_strength[1]))
            img += xp.asarray(bg, dtype=dtype)
        elif background == "lowfreq":
            tex = _lowfreq_texture(rng, H, W, dtype=dtype, block=max(2, H // 8))
            bg0 = float(rng.uniform(bg_strength[0], bg_strength[1]))
            img += (xp.asarray(bg0, dtype=dtype) * tex)[None, :, :]
        elif background == "none":
            pass
        else:
            raise ValueError('background must be "none", "uniform", or "lowfreq"')

        # Main shape params
        cy = base_cy + float(rng.uniform(-max_shift, max_shift))
        cx = base_cx + float(rng.uniform(-max_shift, max_shift))
        size = float(rng.uniform(s_lo, s_hi))
        theta = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0

        # Color
        if channels == 1:
            color = xp.asarray([float(rng.uniform(0.6, 1.0))], dtype=dtype)
        else:
            color = rng.uniform(0.4, 1.0, size=(3,)).astype(dtype)

        label = int(y[i])
        if label == 0:
            mask = _circle_mask(yy, xx, cy, cx, r=size * 0.5)
        elif label == 1:
            half = size * 0.5
            mask = _rotated_rect_mask(yy, xx, cy, cx, half_h=half, half_w=half, theta=theta)
        else:
            mask, _ = _triangle_mask(yy, xx, cy, cx, size=size * 0.6, theta=theta, dtype=dtype)

        img = _apply_color(img, mask, color)

        # Clutter distractors
        for _ in range(int(clutter)):
            ccy = float(rng.uniform(0.0, H - 1.0))
            ccx = float(rng.uniform(0.0, W - 1.0))
            csize = float(rng.uniform(c_lo, c_hi))
            ctheta = float(rng.uniform(0.0, 2.0 * math.pi))

            if channels == 1:
                ccolor = xp.asarray([float(rng.uniform(0.2, 0.8))], dtype=dtype)
            else:
                ccolor = rng.uniform(0.1, 0.9, size=(3,)).astype(dtype)

            kind = int(rng.randint(0, 3))
            if kind == 0:
                cm = _circle_mask(yy, xx, ccy, ccx, r=csize * 0.5)
            elif kind == 1:
                half = csize * 0.5
                cm = _rotated_rect_mask(yy, xx, ccy, ccx, half_h=half, half_w=half, theta=ctheta)
            else:
                cm, _ = _triangle_mask(yy, xx, ccy, ccx, size=csize * 0.6, theta=ctheta, dtype=dtype)

            img = _apply_color(img, cm, ccolor)

        # Shortcut watermark
        correct_corner = watermark_corners[label]
        if float(rng.uniform(0.0, 1.0)) < float(p_shortcut):
            corner = correct_corner
        else:
            # wrong corner: pick one of the other class corners
            other = [c for c in watermark_corners if c != correct_corner]
            corner = other[int(rng.randint(0, len(other)))]

        inten = float(rng.uniform(float(watermark_intensity[0]), float(watermark_intensity[1])))
        img = _add_watermark(img, corner=corner, size=watermark_size, intensity=inten, channels=channels)

        # Pixel noise
        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

        img = xp.clip(img, 0.0, 1.0)
        X[i] = img

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y