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


def make_shapes_detection_occluded(
    n_samples=500,
    image_size=64,
    channels=1,              # NCHW
    min_objects=3,
    max_objects=8,
    shape_scale=(0.14, 0.30),
    rotation=True,

    # crowding / occlusion controls
    crowd_radius=0.18,       # fraction of image_size around a cluster center where objects are placed
    min_iou=0.15,            # enforce some overlap (>=)
    max_iou=0.85,            # avoid complete nesting (<=)
    max_tries=200,

    background="lowfreq",
    bg_strength=(0.0, 0.30),
    noise_std=0.03,

    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Detection dataset with forced overlap/occlusion.
    Output: list of dicts:
      {
        "image": (C,H,W),
        "boxes": (N,4) xyxy,
        "labels": (N,) 0=circle 1=square 2=triangle
      }
    Note: boxes are for the full shapes (not the visible part). Occlusion is natural from draw order.
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if min_objects < 0 or max_objects < min_objects:
        raise ValueError("invalid min_objects/max_objects")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    min_dim = float(min(H, W))
    s_lo = float(shape_scale[0]) * min_dim
    s_hi = float(shape_scale[1]) * min_dim

    # cluster center for crowding
    cc_y = float(rng.uniform(0.35 * (H - 1), 0.65 * (H - 1)))
    cc_x = float(rng.uniform(0.35 * (W - 1), 0.65 * (W - 1)))
    rad = float(crowd_radius) * min_dim

    samples = []

    for _ in range(n_samples):
        img = xp.zeros((channels, H, W), dtype=dtype)

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

        boxes = []
        labels = []
        masks = []  # not returned, used to compute visible if you want later

        # place objects with overlap constraints
        for _k in range(n_obj):
            placed = False
            for _t in range(int(max_tries)):
                label = int(rng.randint(0, 3))
                size = float(rng.uniform(s_lo, s_hi))
                theta = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0

                # sample center near crowd center
                cx = float(cc_x + rng.normal(0.0, rad))
                cy = float(cc_y + rng.normal(0.0, rad))

                # keep within frame
                pad = size * 0.7 + 1.0
                cx = float(xp.clip(cx, pad, W - 1.0 - pad))
                cy = float(xp.clip(cy, pad, H - 1.0 - pad))

                # compute mask + bbox
                if label == 0:
                    r = size * 0.5
                    mask = _circle_mask(yy, xx, cy, cx, r=r)
                    box = _bbox_from_circle(cx, cy, r, W, H, dtype)
                elif label == 1:
                    half = size * 0.5
                    mask = _rotated_rect_mask(yy, xx, cy, cx, half_h=half, half_w=half, theta=theta)
                    box = _bbox_from_rotated_square(cx, cy, half, theta, W, H, dtype)
                else:
                    mask, box = _triangle_mask_and_bbox(yy, xx, cy, cx, size=size * 0.6, theta=theta, W=W, H=H, dtype=dtype)

                # enforce overlap band with at least one existing object (except the first)
                if boxes:
                    ious = [float(_iou_xyxy(box, b, dtype)) for b in boxes]
                    mx = max(ious)
                    if not (mx >= float(min_iou) and mx <= float(max_iou)):
                        continue

                placed = True
                boxes.append(box)
                labels.append(label)
                masks.append(mask)
                break

            if not placed:
                # couldn't satisfy overlap constraints, just skip this object
                pass

        # draw in random order to create occlusion
        order = rng.permutation(len(boxes)) if len(boxes) > 0 else []
        for idx in (order.tolist() if hasattr(order, "tolist") else order):
            label = int(labels[idx])
            mask = masks[idx]

            if channels == 1:
                color = xp.asarray([float(rng.uniform(0.55, 1.0))], dtype=dtype)
            else:
                color = rng.uniform(0.35, 1.0, size=(3,)).astype(dtype)

            img = _apply_color(img, mask, color)

        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

        img = xp.clip(img, 0.0, 1.0)

        if boxes:
            boxes_arr = xp.stack(boxes, axis=0).astype(dtype)
            labels_arr = xp.asarray(labels, dtype=xp.int64)
        else:
            boxes_arr = xp.zeros((0, 4), dtype=dtype)
            labels_arr = xp.zeros((0,), dtype=xp.int64)

        samples.append({"image": img.astype(dtype), "boxes": boxes_arr, "labels": labels_arr})

    if shuffle:
        perm = _get_rng(random_state).permutation(len(samples))
        samples = [samples[i] for i in perm.tolist()]

    return samples