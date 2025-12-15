import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _make_grid,
                                             _apply_color,
                                             _circle_mask,
                                             _rotated_rect_mask,
                                             _triangle_mask)
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_attribute_binding(
    n_samples=2000,
    image_size=64,
    channels=3,                 # captions assume colors; 3 is strongly recommended
    shapes=("square", "circle"),# pick 2 distinct shape types per sample
    colors=("red", "blue"),     # pick 2 distinct colors per sample
    include_positions=False,    # if True, captions include "left/right" which makes it stricter
    rotation=True,
    noise_std=0.02,
    background=0.0,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Returns a list of dicts:
      {
        "image": (C,H,W) float [0,1],
        "text_pos": correct caption,
        "text_neg": swapped-attribute caption
      }
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if len(shapes) < 2:
        raise ValueError("need at least 2 shapes")
    if len(colors) < 2:
        raise ValueError("need at least 2 colors")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    # Simple palette (stable names -> stable vectors)
    palette = {
        "red":   xp.asarray([1.0, 0.2, 0.2], dtype=dtype),
        "green": xp.asarray([0.2, 1.0, 0.2], dtype=dtype),
        "blue":  xp.asarray([0.2, 0.3, 1.0], dtype=dtype),
        "yellow":xp.asarray([1.0, 1.0, 0.2], dtype=dtype),
        "white": xp.asarray([1.0, 1.0, 1.0], dtype=dtype),
        "cyan":  xp.asarray([0.2, 1.0, 1.0], dtype=dtype),
        "magenta":xp.asarray([1.0, 0.2, 1.0], dtype=dtype),
    }

    def render_shape(img, shape_name, cy, cx, size, theta, color_vec):
        if shape_name == "circle":
            m = _circle_mask(yy, xx, cy, cx, r=size * 0.5)
        elif shape_name == "square":
            half = size * 0.5
            m = _rotated_rect_mask(yy, xx, cy, cx, half_h=half, half_w=half, theta=theta)
        elif shape_name == "triangle":
            m = _triangle_mask(yy, xx, cy, cx, size=size * 0.6, theta=theta, dtype=dtype)
        else:
            raise ValueError(f"unknown shape '{shape_name}'")
        return _apply_color(img, m, color_vec), m

    samples = []
    min_dim = float(min(H, W))
    size = 0.26 * min_dim  # fixed-ish for consistency
    y_center = (H - 1) / 2.0
    x_left = 0.30 * (W - 1)
    x_right = 0.70 * (W - 1)

    for _ in range(n_samples):
        img = xp.zeros((channels, H, W), dtype=dtype)
        if background and background > 0:
            img += xp.asarray(float(background), dtype=dtype)

        # pick 2 distinct shapes and colors
        sh_idx = rng.choice(len(shapes), size=(2,), replace=False).tolist()
        co_idx = rng.choice(len(colors), size=(2,), replace=False).tolist()
        s1, s2 = shapes[int(sh_idx[0])], shapes[int(sh_idx[1])]
        c1n, c2n = colors[int(co_idx[0])], colors[int(co_idx[1])]

        t1 = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0
        t2 = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0

        if channels == 1:
            c1 = xp.asarray([0.9], dtype=dtype)
            c2 = xp.asarray([0.6], dtype=dtype)
        else:
            c1 = palette.get(c1n, palette["red"])
            c2 = palette.get(c2n, palette["blue"])

        # draw left then right (order fixed to avoid extra confounds)
        img, _ = render_shape(img, s1, y_center, x_left,  size, t1, c1)
        img, _ = render_shape(img, s2, y_center, x_right, size, t2, c2)

        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)
        img = xp.clip(img, 0.0, 1.0)

        if include_positions:
            text_pos = f"a {c1n} {s1} on the left and a {c2n} {s2} on the right"
            text_neg = f"a {c2n} {s1} on the left and a {c1n} {s2} on the right"
        else:
            text_pos = f"a {c1n} {s1} and a {c2n} {s2}"
            text_neg = f"a {c2n} {s1} and a {c1n} {s2}"

        samples.append({"image": img.astype(dtype), "text_pos": text_pos, "text_neg": text_neg})

    if shuffle:
        perm = rng.permutation(len(samples)).tolist()
        samples = [samples[i] for i in perm]

    return samples
