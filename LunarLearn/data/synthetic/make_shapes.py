import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import _get_rng, _make_grid, _lowfreq_texture, _apply_color, _circle_mask, _rotated_rect_mask, _triangle_mask
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_shapes(
    n_samples=2000,
    image_size=32,
    channels=1,                 # 1 or 3
    n_classes=3,                # 3 => circle/square/triangle
    shape_scale=(0.25, 0.55),   # relative to min(H,W)
    rotation=True,
    translate=0.25,             # fraction of image_size for random center shift
    background="lowfreq",       # "none" | "uniform" | "lowfreq"
    bg_strength=(0.0, 0.35),    # intensity range for background
    clutter=2,                  # number of distractor shapes per image
    clutter_scale=(0.08, 0.20),
    noise_std=0.05,
    flip_y=0.0,                 # label noise
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Procedural shapes classification dataset.
    Labels:
        0: circle
        1: square
        2: triangle
    Returns:
        X: (n_samples, C, H, W) in [0,1]
        y: (n_samples,) int64
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if n_classes != 3:
        raise ValueError("n_classes must be 3 (circle/square/triangle) for now")
    if H < 8:
        raise ValueError("image_size too small to be useful")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    X = xp.zeros((n_samples, channels, H, W), dtype=dtype)
    y = rng.randint(0, 3, size=(n_samples,), dtype=xp.int64)

    # label noise
    if flip_y and flip_y > 0:
        n_flip = int(round(float(flip_y) * n_samples))
        if n_flip > 0:
            idx = rng.choice(n_samples, size=(n_flip,), replace=False)
            new_y = rng.randint(0, 3, size=(n_flip,), dtype=xp.int64)
            same = new_y == y[idx]
            if xp.any(same):
                new_y[same] = (new_y[same] + 1) % 3
            y[idx] = new_y

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

        # Background (write into CHW)
        if background == "uniform":
            bg = float(rng.uniform(bg_strength[0], bg_strength[1]))
            img += xp.asarray(bg, dtype=dtype)
        elif background == "lowfreq":
            tex = _lowfreq_texture(rng, H, W, dtype=dtype, block=max(2, H // 8))  # (H,W)
            bg0 = float(rng.uniform(bg_strength[0], bg_strength[1]))
            img += (xp.asarray(bg0, dtype=dtype) * tex)[None, :, :]  # (1,H,W) broadcast to C
        elif background == "none":
            pass
        else:
            raise ValueError('background must be "none", "uniform", or "lowfreq"')

        # Main shape parameters
        cy = base_cy + float(rng.uniform(-max_shift, max_shift))
        cx = base_cx + float(rng.uniform(-max_shift, max_shift))
        size = float(rng.uniform(s_lo, s_hi))
        theta = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0

        # Shape color (C,)
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
            mask = _triangle_mask(yy, xx, cy, cx, size=size * 0.6, theta=theta)

        img = _apply_color(img, mask, color)

        # Clutter
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
                cm = _triangle_mask(yy, xx, ccy, ccx, size=csize * 0.6, theta=ctheta)
                
            img = _apply_color(img, cm, ccolor)

        # Pixel noise
        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, noise_std, size=img.shape).astype(dtype)

        X[i] = xp.clip(img, 0.0, 1.0)

    if shuffle:
        perm = rng.permutation(n_samples)
        X = X[perm]
        y = y[perm]

    return X.astype(dtype), y