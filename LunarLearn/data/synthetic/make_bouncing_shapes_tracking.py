import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _circle_mask,
                                             _make_grid,
                                             _apply_color,
                                             _bbox_from_center,
                                             _square_mask,
                                             _resolve_wall_bounce,
                                             _resolve_pair_collision)
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_bouncing_shapes_tracking(
    n_sequences=200,
    seq_len=30,
    frame_size=64,
    channels=1,               # NCHW
    n_objects=5,              # fixed K for tracking
    shapes=("circle", "square"),
    radius_range=(5, 10),
    speed_range=(1.0, 3.0),
    background=0.0,
    noise_std=0.0,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Returns:
      video: (N,T,C,H,W)
      boxes: (N,T,K,4) xyxy
      track_ids: (K,) int64
      visible: (N,T,K) int64 (1 for all here)
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(frame_size)
    T = int(seq_len)
    K = int(n_objects)
    if K < 1:
        raise ValueError("n_objects must be >= 1")
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    if channels == 1:
        palette = [xp.asarray([0.75], dtype=dtype), xp.asarray([0.95], dtype=dtype), xp.asarray([0.55], dtype=dtype)]
    else:
        palette = [
            xp.asarray([1.0, 0.2, 0.2], dtype=dtype),
            xp.asarray([0.2, 1.0, 0.2], dtype=dtype),
            xp.asarray([0.2, 0.3, 1.0], dtype=dtype),
            xp.asarray([1.0, 1.0, 0.2], dtype=dtype),
        ]

    video = xp.zeros((n_sequences, T, channels, H, W), dtype=dtype)
    boxes = xp.zeros((n_sequences, T, K, 4), dtype=dtype)
    visible = xp.ones((n_sequences, T, K), dtype=xp.int64)
    track_ids = xp.arange(K, dtype=xp.int64)

    def render_obj(img, obj):
        r = xp.asarray(obj["r"], dtype=dtype)
        cy = xp.asarray(obj["y"], dtype=dtype)
        cx = xp.asarray(obj["x"], dtype=dtype)
        if obj["shape"] == "circle":
            m = _circle_mask(yy, xx, cy, cx, r=r)
        else:
            m = _square_mask(yy, xx, cy, cx, r=r)
        return _apply_color(img, m, obj["color"])

    for n in range(n_sequences):
        objs = []
        for k in range(K):
            r = float(rng.uniform(float(radius_range[0]), float(radius_range[1])))
            x = float(rng.uniform(r, (W - 1) - r))
            y = float(rng.uniform(r, (H - 1) - r))
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            spd = float(rng.uniform(float(speed_range[0]), float(speed_range[1])))
            vx = spd * math.cos(ang)
            vy = spd * math.sin(ang)
            shape = shapes[int(rng.randint(0, len(shapes)))]
            color = palette[int(rng.randint(0, len(palette)))]
            objs.append({"id": k, "x": x, "y": y, "vx": vx, "vy": vy, "r": r, "shape": shape, "color": color})

        for t in range(T):
            img = xp.zeros((channels, H, W), dtype=dtype)
            if background and background != 0.0:
                img += xp.asarray(float(background), dtype=dtype)

            for obj in objs:
                img = render_obj(img, obj)
                boxes[n, t, obj["id"]] = _bbox_from_center(obj["x"], obj["y"], obj["r"], W, H, dtype)

            if noise_std and noise_std > 0:
                img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)
            video[n, t] = xp.clip(img, 0.0, 1.0)

            if t == T - 1:
                break

            # step
            for obj in objs:
                obj["x"] += obj["vx"]
                obj["y"] += obj["vy"]
                obj["x"], obj["y"], obj["vx"], obj["vy"] = _resolve_wall_bounce(
                    obj["x"], obj["y"], obj["vx"], obj["vy"], obj["r"], W, H
                )

            # collisions
            for i in range(K):
                for j in range(i + 1, K):
                    _resolve_pair_collision(objs[i], objs[j])

    if shuffle:
        perm = rng.permutation(n_sequences)
        video = video[perm]
        boxes = boxes[perm]
        visible = visible[perm]

    return video.astype(dtype), boxes.astype(dtype), track_ids, visible