import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _make_grid,
                                             _circle_mask,
                                             _rotated_rect_mask,
                                             _triangle_mask,
                                             _apply_color,
                                             _lowfreq_texture)
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_shapes_caption(
    n_samples=2000,
    image_size=64,
    channels=3,                  # NCHW output; captions assume colors so 3 is sensible
    n_objects_range=(1, 2),       # 1 or 2 objects (keeps captions simple and reliable)
    shape_scale=(0.18, 0.32),
    rotation=True,
    background="lowfreq",         # "none" | "uniform" | "lowfreq"
    bg_strength=(0.0, 0.25),
    noise_std=0.02,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Shapes + Caption dataset.
    Output: list of dicts: {"image": (C,H,W) float [0,1], "text": caption string}

    Captions are programmatic, e.g.:
      - "a red circle"
      - "a red circle left of a blue square"
      - "two triangles" (only if same shape and 2 objects)
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    # fixed color palette for stable text
    palette = [
        ("red",   xp.asarray([1.0, 0.2, 0.2], dtype=dtype)),
        ("green", xp.asarray([0.2, 1.0, 0.2], dtype=dtype)),
        ("blue",  xp.asarray([0.2, 0.3, 1.0], dtype=dtype)),
        ("yellow",xp.asarray([1.0, 1.0, 0.2], dtype=dtype)),
    ]
    shapes = ["circle", "square", "triangle"]

    samples = []

    min_dim = float(min(H, W))
    s_lo = float(shape_scale[0]) * min_dim
    s_hi = float(shape_scale[1]) * min_dim

    def draw_shape(img, shape_id, cy, cx, size, theta, color_vec):
        if shape_id == 0:
            mask = _circle_mask(yy, xx, cy, cx, r=size * 0.5)
        elif shape_id == 1:
            half = size * 0.5
            mask = _rotated_rect_mask(yy, xx, cy, cx, half_h=half, half_w=half, theta=theta)
        else:
            mask = _triangle_mask(yy, xx, cy, cx, size=size * 0.6, theta=theta, dtype=dtype)
        return _apply_color(img, mask, color_vec), mask

    for _ in range(n_samples):
        img = xp.zeros((channels, H, W), dtype=dtype)

        # Background
        if background == "uniform":
            bg = float(rng.uniform(bg_strength[0], bg_strength[1]))
            img += xp.asarray(bg, dtype=dtype)
        elif background == "lowfreq":
            tex = _lowfreq_texture(rng, H, W, dtype=dtype, block=max(2, H // 10))
            bg0 = float(rng.uniform(bg_strength[0], bg_strength[1]))
            img += (xp.asarray(bg0, dtype=dtype) * tex)[None, :, :]
        elif background == "none":
            pass
        else:
            raise ValueError('background must be "none", "uniform", or "lowfreq"')

        n_obj = int(rng.randint(int(n_objects_range[0]), int(n_objects_range[1]) + 1))

        objs = []
        for k in range(n_obj):
            shape_id = int(rng.randint(0, 3))
            cname, cvec = palette[int(rng.randint(0, len(palette)))]

            size = float(rng.uniform(s_lo, s_hi))
            theta = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0

            pad = size * 0.8 + 1.0
            cx = float(rng.uniform(pad, W - 1.0 - pad))
            cy = float(rng.uniform(pad, H - 1.0 - pad))

            if channels == 1:
                # grayscale fallback: use intensity; caption still uses color name
                cvec_use = xp.asarray([float(rng.uniform(0.5, 1.0))], dtype=dtype)
            else:
                cvec_use = cvec

            img, _ = draw_shape(img, shape_id, cy, cx, size, theta, cvec_use)
            objs.append({"shape": shapes[shape_id], "color": cname, "cx": cx, "cy": cy})

        # Caption
        if len(objs) == 1:
            o = objs[0]
            text = f"a {o['color']} {o['shape']}"
        else:
            a, b = objs[0], objs[1]

            # If same shape and same color, allow counting caption
            if a["shape"] == b["shape"] and a["color"] == b["color"]:
                text = f"two {a['color']} {a['shape']}s"
            else:
                # relative position
                dx = a["cx"] - b["cx"]
                dy = a["cy"] - b["cy"]

                if abs(dx) >= abs(dy):
                    rel = "left of" if dx < 0 else "right of"
                else:
                    rel = "above" if dy < 0 else "below"

                text = f"a {a['color']} {a['shape']} {rel} a {b['color']} {b['shape']}"

        # Noise + clip
        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

        img = xp.clip(img, 0.0, 1.0)

        samples.append({"image": img.astype(dtype), "text": text})

    if shuffle:
        perm = rng.permutation(len(samples)).tolist()
        samples = [samples[i] for i in perm]

    return samples