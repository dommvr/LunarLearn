import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _make_grid,
                                             _lowfreq_texture,
                                             _apply_color,
                                             _circle_mask,
                                             _rotated_rect_mask,
                                             _bbox_from_circle,
                                             _bbox_from_rotated_square,
                                             _triangle_mask_and_bbox,
                                             _iou_xyxy)
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_shapes_segmentation(
    n_samples=1000,
    image_size=64,
    channels=1,                 # NCHW
    min_objects=1,
    max_objects=5,
    shape_scale=(0.12, 0.32),
    rotation=True,
    background="lowfreq",
    bg_strength=(0.0, 0.30),
    noise_std=0.03,
    overlap=True,               # if False, tries to reduce overlap
    max_tries=100,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Semantic segmentation:
      mask values:
        0 background
        1 circle
        2 square
        3 triangle

    Output per sample:
      {"image": (C,H,W), "mask": (H,W) int64}
    Returns: list of dicts (variable content per sample is easy this way)
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    min_dim = float(min(H, W))
    s_lo = float(shape_scale[0]) * min_dim
    s_hi = float(shape_scale[1]) * min_dim

    samples = []

    for _ in range(n_samples):
        img = xp.zeros((channels, H, W), dtype=dtype)
        mask_out = xp.zeros((H, W), dtype=xp.int64)

        # background
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

        n_obj = int(rng.randint(int(min_objects), int(max_objects) + 1))

        placed_boxes = []

        # draw objects in random order; last drawn wins in both image + mask (occlusion)
        for _k in range(n_obj):
            for _t in range(int(max_tries)):
                label = int(rng.randint(0, 3))
                size = float(rng.uniform(s_lo, s_hi))
                theta = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0

                pad = size * 0.7 + 1.0
                cx = float(rng.uniform(pad, W - 1.0 - pad))
                cy = float(rng.uniform(pad, H - 1.0 - pad))

                if label == 0:
                    r = size * 0.5
                    shape_mask = _circle_mask(yy, xx, cy, cx, r=r)
                    box = _bbox_from_circle(cx, cy, r, W, H, dtype)
                elif label == 1:
                    half = size * 0.5
                    shape_mask = _rotated_rect_mask(yy, xx, cy, cx, half_h=half, half_w=half, theta=theta)
                    box = _bbox_from_rotated_square(cx, cy, half, theta, W, H, dtype)
                else:
                    shape_mask, box = _triangle_mask_and_bbox(yy, xx, cy, cx, size=size * 0.6, theta=theta, W=W, H=H, dtype=dtype)

                if not overlap and placed_boxes:
                    # crude "avoid overlap": reject if IoU too high
                    ious = [float(_iou_xyxy(box, b, dtype)) for b in placed_boxes]
                    if max(ious) > 0.05:
                        continue

                placed_boxes.append(box)

                # draw color
                if channels == 1:
                    color = xp.asarray([float(rng.uniform(0.55, 1.0))], dtype=dtype)
                else:
                    color = rng.uniform(0.35, 1.0, size=(3,)).astype(dtype)

                img = _apply_color(img, shape_mask, color)

                # semantic mask: 1..3
                cls_id = int(label + 1)
                mask_out = xp.where(shape_mask, xp.asarray(cls_id, dtype=xp.int64), mask_out)

                break  # placed this object

        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

        img = xp.clip(img, 0.0, 1.0)
        samples.append({"image": img.astype(dtype), "mask": mask_out})

    if shuffle:
        perm = _get_rng(random_state).permutation(len(samples))
        samples = [samples[i] for i in perm.tolist()]

    return samples