import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _make_grid,
                                             _lowfreq_texture,
                                             _apply_color,
                                             _circle_mask,
                                             _rotated_rect_mask,
                                             _triangle_mask,
                                             _bbox_from_circle,
                                             _bbox_from_rotated_square,
                                             _bbox_from_vertices)
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_shapes_detection(
    n_samples=500,
    image_size=64,
    channels=1,                  # NCHW output
    min_objects=1,
    max_objects=5,
    shape_scale=(0.12, 0.30),
    rotation=True,
    background="lowfreq",        # "none" | "uniform" | "lowfreq"
    bg_strength=(0.0, 0.35),
    noise_std=0.03,

    # optional: avoid extreme overlap
    max_tries=50,
    min_iou_avoid=0.0,           # 0 => allow overlap; >0 avoids placing shapes with IoU > threshold

    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Shapes detection dataset. Per sample output:
      {
        "image": (C,H,W) float in [0,1],
        "boxes": (N,4) float xyxy in pixel coords,
        "labels": (N,) int64 (0 circle, 1 square, 2 triangle)
      }

    Returns:
      samples: list of dicts length n_samples
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

    def iou_xyxy(a, b):
        # a,b: (4,) xyxy in pixel coords
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = xp.maximum(ax1, bx1)
        iy1 = xp.maximum(ay1, by1)
        ix2 = xp.minimum(ax2, bx2)
        iy2 = xp.minimum(ay2, by2)
        iw = xp.maximum(ix2 - ix1, 0.0)
        ih = xp.maximum(iy2 - iy1, 0.0)
        inter = iw * ih
        area_a = xp.maximum(ax2 - ax1, 0.0) * xp.maximum(ay2 - ay1, 0.0)
        area_b = xp.maximum(bx2 - bx1, 0.0) * xp.maximum(by2 - by1, 0.0)
        return inter / (area_a + area_b - inter + xp.asarray(1e-12, dtype=dtype))

    samples = []

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

        n_obj = int(rng.randint(int(min_objects), int(max_objects) + 1))
        boxes = []
        labels = []

        for _k in range(n_obj):
            placed = False
            for _try in range(int(max_tries)):
                label = int(rng.randint(0, 3))
                size = float(rng.uniform(s_lo, s_hi))
                theta = float(rng.uniform(0.0, 2.0 * math.pi)) if rotation else 0.0

                # sample center with padding so it doesn't go out of frame too much
                pad = size * 0.6 + 1.0
                cx = float(rng.uniform(pad, W - 1.0 - pad))
                cy = float(rng.uniform(pad, H - 1.0 - pad))

                # compute bbox + mask
                if label == 0:
                    r = size * 0.5
                    mask = _circle_mask(yy, xx, cy, cx, r=r)
                    box = _bbox_from_circle(cx, cy, r, W, H, dtype)
                elif label == 1:
                    half = size * 0.5
                    mask = _rotated_rect_mask(yy, xx, cy, cx, half_h=half, half_w=half, theta=theta)
                    box = _bbox_from_rotated_square(cx, cy, half, theta, W, H, dtype)
                else:
                    mask, V = _triangle_mask(yy, xx, cy, cx, size=size * 0.6, theta=theta, dtype=dtype)
                    box = _bbox_from_vertices(V, W, H, dtype)

                # overlap avoidance
                if min_iou_avoid and min_iou_avoid > 0 and boxes:
                    ok = True
                    for b in boxes:
                        if float(iou_xyxy(box, b)) > float(min_iou_avoid):
                            ok = False
                            break
                    if not ok:
                        continue

                # draw it
                if channels == 1:
                    color = xp.asarray([float(rng.uniform(0.55, 1.0))], dtype=dtype)
                else:
                    color = rng.uniform(0.35, 1.0, size=(3,)).astype(dtype)

                img = _apply_color(img, mask, color)

                boxes.append(box)
                labels.append(label)
                placed = True
                break

            if not placed:
                # give up on this object; dataset still valid
                pass

        if noise_std and noise_std > 0:
            img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

        img = xp.clip(img, 0.0, 1.0)

        if boxes:
            boxes_arr = xp.stack(boxes, axis=0).astype(dtype)  # (N,4)
            labels_arr = xp.asarray(labels, dtype=xp.int64)
        else:
            boxes_arr = xp.zeros((0, 4), dtype=dtype)
            labels_arr = xp.zeros((0,), dtype=xp.int64)

        samples.append({"image": img.astype(dtype), "boxes": boxes_arr, "labels": labels_arr})

    if shuffle:
        perm = _get_rng(random_state).permutation(len(samples))
        samples = [samples[i] for i in perm.tolist()]  # perm is xp array; tolist ok at small n

    return samples