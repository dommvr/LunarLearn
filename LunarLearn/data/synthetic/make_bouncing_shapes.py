import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _circle_mask,
                                             _make_grid,
                                             _apply_color,
                                             _triangle_mask,
                                             _square_mask,
                                             _resolve_wall_bounce,
                                             _resolve_pair_collision)
import math

xp = backend.xp
DTYPE = backend.DTYPE


def make_bouncing_shapes(
    n_sequences=500,
    seq_len=20,
    frame_size=64,
    channels=1,                    # NCHW
    n_objects_range=(2, 5),
    shapes=("circle", "square", "triangle"),
    radius_range=(5, 10),          # pixels
    speed_range=(1.0, 3.0),        # pixels per step
    background=0.0,
    noise_std=0.0,

    # events
    merge_prob=0.0,                # if >0, collisions sometimes merge (event=2)
    merge_growth=1.15,             # radius multiplier for merged object

    # outputs
    return_events=True,            # events are per transition t->t+1 (length T-1)
    return_next_frame_pairs=False,

    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Physics-lite bouncing shapes with elastic collisions.

    Returns:
      video: (N, T, C, H, W)
      optionally:
        events: (N, T-1) int64 where 0=no event, 1=collision, 2=merge
        X_next, Y_next: (N, T-1, C, H, W) for next-frame prediction
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(frame_size)
    T = int(seq_len)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if T < 2:
        raise ValueError("seq_len must be >= 2")
    if n_objects_range[0] < 1 or n_objects_range[1] < n_objects_range[0]:
        raise ValueError("invalid n_objects_range")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    # simple stable palette
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
    events = xp.zeros((n_sequences, T - 1), dtype=xp.int64) if return_events else None

    def render_obj(img, obj):
        r = xp.asarray(obj["r"], dtype=dtype)
        cy = xp.asarray(obj["y"], dtype=dtype)
        cx = xp.asarray(obj["x"], dtype=dtype)
        shp = obj["shape"]
        if shp == "circle":
            m = _circle_mask(yy, xx, cy, cx, r=r)
        elif shp == "square":
            m = _square_mask(yy, xx, cy, cx, r=r)
        elif shp == "triangle":
            m = _triangle_mask(yy, xx, cy, cx, r=r, dtype=dtype)
        else:
            raise ValueError(f"unknown shape {shp}")
        return _apply_color(img, m, obj["color"])

    for n in range(n_sequences):
        n_obj = int(rng.randint(int(n_objects_range[0]), int(n_objects_range[1]) + 1))

        # init objects
        objs = []
        for k in range(n_obj):
            r = float(rng.uniform(float(radius_range[0]), float(radius_range[1])))
            # center inside bounds
            x = float(rng.uniform(r, (W - 1) - r))
            y = float(rng.uniform(r, (H - 1) - r))

            # random velocity direction + magnitude
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            spd = float(rng.uniform(float(speed_range[0]), float(speed_range[1])))
            vx = spd * math.cos(ang)
            vy = spd * math.sin(ang)

            shape = shapes[int(rng.randint(0, len(shapes)))]
            color = palette[int(rng.randint(0, len(palette)))]

            objs.append({"id": k, "x": x, "y": y, "vx": vx, "vy": vy, "r": r, "shape": shape, "color": color})

        # simulate frames
        for t in range(T):
            img = xp.zeros((channels, H, W), dtype=dtype)
            if background and background != 0.0:
                img += xp.asarray(float(background), dtype=dtype)

            # draw (fixed order; occlusion is stable)
            for obj in objs:
                img = render_obj(img, obj)

            if noise_std and noise_std > 0:
                img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)
            img = xp.clip(img, 0.0, 1.0)

            video[n, t] = img

            if t == T - 1:
                break

            # step positions
            for obj in objs:
                obj["x"] += obj["vx"]
                obj["y"] += obj["vy"]
                obj["x"], obj["y"], obj["vx"], obj["vy"] = _resolve_wall_bounce(
                    obj["x"], obj["y"], obj["vx"], obj["vy"], obj["r"], W, H
                )

            # resolve collisions (pairwise)
            any_collision = False
            any_merge = False

            i = 0
            while i < len(objs):
                j = i + 1
                while j < len(objs):
                    hit = _resolve_pair_collision(objs[i], objs[j])
                    if hit:
                        any_collision = True
                        # optional merge event
                        if merge_prob and float(rng.uniform(0.0, 1.0)) < float(merge_prob) and len(objs) > 1:
                            # merge j into i (keep i's ID)
                            objs[i]["r"] = float(objs[i]["r"]) * float(merge_growth)
                            # average velocities (very fake physics, but stable)
                            objs[i]["vx"] = 0.5 * (objs[i]["vx"] + objs[j]["vx"])
                            objs[i]["vy"] = 0.5 * (objs[i]["vy"] + objs[j]["vy"])
                            # remove j
                            objs.pop(j)
                            any_merge = True
                            continue  # don't increment j, list shifted
                    j += 1
                i += 1

            if return_events:
                if any_merge:
                    events[n, t] = 2
                elif any_collision:
                    events[n, t] = 1
                else:
                    events[n, t] = 0

    if shuffle:
        perm = rng.permutation(n_sequences)
        video = video[perm]
        if events is not None:
            events = events[perm]

    if return_next_frame_pairs:
        X_next = video[:, :-1]
        Y_next = video[:, 1:]
        if return_events:
            return video.astype(dtype), X_next.astype(dtype), Y_next.astype(dtype), events
        return video.astype(dtype), X_next.astype(dtype), Y_next.astype(dtype)

    if return_events:
        return video.astype(dtype), events
    return video.astype(dtype)