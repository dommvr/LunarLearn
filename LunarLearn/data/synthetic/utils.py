import LunarLearn.core.backend.backend as backend
import math

xp = backend.xp
SEED = backend.SEED


def _get_rng(random_state):
    """
    Return a RandomState-like object for both numpy and cupy.
    """
    if random_state is None:
        random_state = SEED
    # numpy.random.RandomState, cupy.random.RandomState both exist
    return xp.random.RandomState(int(random_state))


def _normalize_weights(weights, n_classes):
    if weights is None:
        return None
    if len(weights) != n_classes:
        raise ValueError(f"class_weights must have length {n_classes}, got {len(weights)}")
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("class_weights must sum to a positive number")
    return [w / s for w in weights]


def _allocate_counts(n_samples, weights):
    """
    Allocate integer counts summing to n_samples according to weights.
    """
    n_classes = len(weights)
    raw = [w * n_samples for w in weights]
    counts = [int(x) for x in raw]
    rem = n_samples - sum(counts)

    # distribute remainder to largest fractional parts
    fracs = [(raw[i] - counts[i], i) for i in range(n_classes)]
    fracs.sort(reverse=True)
    for k in range(rem):
        counts[fracs[k][1]] += 1
    return counts


def _flip_labels(y, n_classes, flip_y, rng):
    """Flip a fraction of labels to a random *different* class."""
    if flip_y is None or flip_y <= 0:
        return y
    n = int(round(float(flip_y) * y.shape[0]))
    if n <= 0:
        return y

    idx = rng.choice(y.shape[0], size=(n,), replace=False)
    new_y = rng.randint(0, n_classes, size=(n,), dtype=xp.int64)

    same = new_y == y[idx]
    if xp.any(same):
        new_y[same] = (new_y[same] + 1) % n_classes

    y[idx] = new_y
    return y


def _sample_values(rng, k, distribution, dtype, nonnegative=True):
    """
    Generate positive-ish values to mimic counts/TF-IDF.
    """
    if distribution == "poisson":
        # integer-ish counts (>=0)
        v = rng.poisson(lam=1.0, size=(k,)).astype(dtype)
    elif distribution == "exponential":
        v = rng.exponential(scale=1.0, size=(k,)).astype(dtype)
    elif distribution == "lognormal":
        v = rng.lognormal(mean=0.0, sigma=1.0, size=(k,)).astype(dtype)
    elif distribution == "normal":
        v = rng.normal(loc=0.0, scale=1.0, size=(k,)).astype(dtype)
    else:
        raise ValueError("distribution must be one of: poisson, exponential, lognormal, normal")

    if nonnegative:
        v = xp.maximum(v, xp.asarray(0, dtype=dtype))
    return v


def _make_grid(H, W, dtype):
    ys = xp.arange(H, dtype=dtype)
    xs = xp.arange(W, dtype=dtype)
    yy, xx = xp.meshgrid(ys, xs, indexing="ij")  # (H,W), (H,W)
    return yy, xx


def _lowfreq_texture(rng, H, W, dtype, block=8):
    """
    Blocky low-frequency noise texture in [0,1].
    """
    block = int(block)
    block = max(1, block)

    h0 = max(1, H // block)
    w0 = max(1, W // block)
    small = rng.uniform(0.0, 1.0, size=(h0, w0)).astype(dtype)

    # Upsample by repeat (nearest)
    tex = xp.repeat(xp.repeat(small, block, axis=0), block, axis=1)
    # Crop/pad to exact size
    tex = tex[:H, :W]
    if tex.shape[0] < H or tex.shape[1] < W:
        out = xp.zeros((H, W), dtype=dtype)
        out[: tex.shape[0], : tex.shape[1]] = tex
        tex = out
    return tex


def _apply_color(img, mask, color):
    # img: (C,H,W), mask: (H,W) bool, color: (C,)
    for c in range(img.shape[0]):
        ch = img[c]
        ch[mask] = xp.maximum(ch[mask], color[c])
        img[c] = ch
    return img


def _circle_mask(yy, xx, cy, cx, r):
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= (r ** 2)


def _rotated_rect_mask(yy, xx, cy, cx, half_h, half_w, theta):
    """
    Rotated rectangle centered at (cy,cx).
    half_h, half_w: half sizes in pixels.
    theta: radians.
    """
    s = xp.sin(theta)
    c = xp.cos(theta)

    y = yy - cy
    x = xx - cx

    # Rotate points into rectangle frame by -theta
    xr = c * x + s * y
    yr = -s * x + c * y

    return (xp.abs(xr) <= half_w) & (xp.abs(yr) <= half_h)


def _point_in_triangle_mask(yy, xx, v0, v1, v2):
    """
    Half-plane test for triangle.
    v0,v1,v2 are (y,x) float coords.
    """
    y = yy
    x = xx

    y0, x0 = v0
    y1, x1 = v1
    y2, x2 = v2

    # Compute signed areas (cross products)
    def edge_fn(ya, xa, yb, xb, yp, xp_):
        return (xp_ - xa) * (yb - ya) - (yp - ya) * (xb - xa)

    e0 = edge_fn(y0, x0, y1, x1, y, x)
    e1 = edge_fn(y1, x1, y2, x2, y, x)
    e2 = edge_fn(y2, x2, y0, x0, y, x)

    # Inside if all non-negative or all non-positive
    has_neg = (e0 < 0) | (e1 < 0) | (e2 < 0)
    has_pos = (e0 > 0) | (e1 > 0) | (e2 > 0)
    return ~(has_neg & has_pos)


def _triangle_vertices(cy, cx, size, theta, dtype):
    # Equilateral triangle around center. Returns (3,2) as (y,x).
    angles = xp.asarray([math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6], dtype=dtype) + theta
    vy = cy + size * xp.sin(angles)
    vx = cx + size * xp.cos(angles)
    return xp.stack([vy, vx], axis=1)  # (3,2)


def _triangle_mask(yy, xx, cy, cx, size, theta, dtype):
    V = _triangle_vertices(cy, cx, size, theta, dtype)
    return _point_in_triangle_mask(yy, xx, V[0], V[1], V[2]), V


def _triangle_mask_and_bbox(yy, xx, cy, cx, size, theta, W, H, dtype):
    V = _triangle_vertices(cy, cx, size, theta, dtype)
    mask = _point_in_triangle_mask(yy, xx, V[0], V[1], V[2])
    ys = V[:, 0]
    xs = V[:, 1]
    x1 = xp.clip(xp.min(xs), 0.0, W - 1.0)
    y1 = xp.clip(xp.min(ys), 0.0, H - 1.0)
    x2 = xp.clip(xp.max(xs), 0.0, W - 1.0)
    y2 = xp.clip(xp.max(ys), 0.0, H - 1.0)
    box = xp.asarray([x1, y1, x2, y2], dtype=dtype)
    return mask, box


def _bbox_from_circle(cx, cy, r, W, H, dtype):
    x1 = xp.clip(cx - r, 0.0, W - 1.0)
    y1 = xp.clip(cy - r, 0.0, H - 1.0)
    x2 = xp.clip(cx + r, 0.0, W - 1.0)
    y2 = xp.clip(cy + r, 0.0, H - 1.0)
    return xp.asarray([x1, y1, x2, y2], dtype=dtype)


def _bbox_from_rotated_square(cx, cy, half, theta, W, H, dtype):
    # corners around origin then rotate
    pts = xp.asarray(
        [[-half, -half], [-half, half], [half, half], [half, -half]],
        dtype=dtype
    )  # (4,2) in (x,y)
    s = xp.sin(theta); c = xp.cos(theta)
    x = pts[:, 0]; y = pts[:, 1]
    xr = c * x - s * y
    yr = s * x + c * y
    xs = xr + cx
    ys = yr + cy
    x1 = xp.clip(xp.min(xs), 0.0, W - 1.0)
    y1 = xp.clip(xp.min(ys), 0.0, H - 1.0)
    x2 = xp.clip(xp.max(xs), 0.0, W - 1.0)
    y2 = xp.clip(xp.max(ys), 0.0, H - 1.0)
    return xp.asarray([x1, y1, x2, y2], dtype=dtype)


def _bbox_from_vertices(V_yx, W, H, dtype):
    ys = V_yx[:, 0]
    xs = V_yx[:, 1]
    x1 = xp.clip(xp.min(xs), 0.0, W - 1.0)
    y1 = xp.clip(xp.min(ys), 0.0, H - 1.0)
    x2 = xp.clip(xp.max(xs), 0.0, W - 1.0)
    y2 = xp.clip(xp.max(ys), 0.0, H - 1.0)
    return xp.asarray([x1, y1, x2, y2], dtype=dtype)


def _bbox_from_center(cx, cy, r, W, H, dtype):
    x1 = xp.clip(cx - r, 0.0, W - 1.0)
    y1 = xp.clip(cy - r, 0.0, H - 1.0)
    x2 = xp.clip(cx + r, 0.0, W - 1.0)
    y2 = xp.clip(cy + r, 0.0, H - 1.0)
    return xp.asarray([x1, y1, x2, y2], dtype=dtype)


def _add_watermark(img, corner, size, intensity, channels):
    # img: (C,H,W)
    C, H, W = img.shape
    s = int(size)
    s = max(1, min(s, H, W))

    if corner == "tl":
        y0, x0 = 0, 0
    elif corner == "tr":
        y0, x0 = 0, W - s
    elif corner == "bl":
        y0, x0 = H - s, 0
    elif corner == "br":
        y0, x0 = H - s, W - s
    else:
        raise ValueError("corner must be one of: tl,tr,bl,br")

    if channels == 1:
        val = xp.asarray([intensity], dtype=img.dtype)
    else:
        # subtle RGB-ish watermark
        val = xp.asarray([intensity, intensity * 0.7, intensity * 0.4], dtype=img.dtype)

    for c in range(C):
        patch = img[c, y0:y0+s, x0:x0+s]
        patch[:] = xp.maximum(patch, val[c])
        img[c, y0:y0+s, x0:x0+s] = patch

    return img


def _iou_xyxy(a, b, dtype):
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


def _upsample_repeat(x, block, H, W):
    # x: (h0, w0) -> nearest upsample by repeat -> crop/pad to (H,W)
    tex = xp.repeat(xp.repeat(x, block, axis=0), block, axis=1)
    tex = tex[:H, :W]
    if tex.shape[0] < H or tex.shape[1] < W:
        out = xp.zeros((H, W), dtype=tex.dtype)
        out[: tex.shape[0], : tex.shape[1]] = tex
        tex = out
    return tex


def _upsample_repeat(x, block, H, W):
    # x: (h0, w0) -> nearest upsample by repeat -> crop/pad to (H,W)
    tex = xp.repeat(xp.repeat(x, block, axis=0), block, axis=1)
    tex = tex[:H, :W]
    if tex.shape[0] < H or tex.shape[1] < W:
        out = xp.zeros((H, W), dtype=tex.dtype)
        out[: tex.shape[0], : tex.shape[1]] = tex
        tex = out
    return tex


def _normalize01(x, dtype):
    eps = xp.asarray(1e-12, dtype=dtype)
    mn = xp.min(x)
    mx = xp.max(x)
    return (x - mn) / (mx - mn + eps)


def _box_blur2d(img2d, iters=1):
    # Simple 3x3 box blur repeated iters times.
    out = img2d
    for _ in range(int(iters)):
        H, W = out.shape
        tmp = xp.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            tmp[0:H,   0:W]   + tmp[0:H,   1:W+1]   + tmp[0:H,   2:W+2] +
            tmp[1:H+1, 0:W]   + tmp[1:H+1, 1:W+1]   + tmp[1:H+1, 2:W+2] +
            tmp[2:H+2, 0:W]   + tmp[2:H+2, 1:W+1]   + tmp[2:H+2, 2:W+2]
        ) / 9.0
    return out


def _render_checkerboard(rng, H, W, dtype, n_squares=None):
    if n_squares is None:
        n_squares = int(rng.randint(2, 10))
    yy, xx = _make_grid(H, W, dtype)
    # Normalize to [0,1)
    yn = yy / xp.asarray(max(H - 1, 1), dtype=dtype)
    xn = xx / xp.asarray(max(W - 1, 1), dtype=dtype)
    yn = xp.clip(yn, 0.0, 1.0 - xp.finfo(dtype).eps)
    xn = xp.clip(xn, 0.0, 1.0 - xp.finfo(dtype).eps)
    i = xp.floor(xn * n_squares).astype(xp.int64)
    j = xp.floor(yn * n_squares).astype(xp.int64)
    pat = ((i + j) % 2).astype(dtype)
    return pat


def _render_stripes(rng, H, W, dtype):
    yy, xx = _make_grid(H, W, dtype)
    yn = yy / xp.asarray(max(H - 1, 1), dtype=dtype)
    xn = xx / xp.asarray(max(W - 1, 1), dtype=dtype)
    yn = xp.clip(yn, 0.0, 1.0 - xp.finfo(dtype).eps)
    xn = xp.clip(xn, 0.0, 1.0 - xp.finfo(dtype).eps)

    ori = int(rng.randint(0, 4))  # 0=h,1=v,2=diag,3=anti
    if ori == 0:
        t = yn
    elif ori == 1:
        t = xn
    elif ori == 2:
        t = 0.5 * (xn + yn)
    else:
        t = 0.5 * (xn - yn) + 0.5

    freq = int(rng.randint(2, 16))                  # periods
    duty = float(rng.uniform(0.15, 0.85))           # thickness as duty cycle
    phase = float(rng.uniform(0.0, 1.0))            # phase jitter

    frac = (xp.asarray(freq, dtype=dtype) * t + xp.asarray(phase, dtype=dtype)) % xp.asarray(1.0, dtype=dtype)
    pat = (frac < xp.asarray(duty, dtype=dtype)).astype(dtype)
    return pat


def _render_gradient(rng, H, W, dtype):
    yy, xx = _make_grid(H, W, dtype)
    yn = yy / xp.asarray(max(H - 1, 1), dtype=dtype)
    xn = xx / xp.asarray(max(W - 1, 1), dtype=dtype)

    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    dx = math.cos(angle)
    dy = math.sin(angle)

    g = xp.asarray(dx, dtype=dtype) * xn + xp.asarray(dy, dtype=dtype) * yn
    g = _normalize01(g, dtype)

    # Optional nonlinearity so it’s not always boring
    p = float(rng.uniform(0.6, 2.0))
    g = xp.power(g, xp.asarray(p, dtype=dtype))
    return g


def _render_perlin_like_blobs(rng, H, W, dtype, octaves=4, base_block=None, blur_iters=1):
    """
    Perlin-ish blobs: sum of multi-scale random grids, upsampled with nearest, then blurred.
    Not true Perlin, but it looks like natural-ish texture and is cheap.
    """
    if base_block is None:
        base_block = max(2, H // 16)

    tex = xp.zeros((H, W), dtype=dtype)
    amp = 1.0
    total_amp = 0.0

    block = int(base_block)
    for _ in range(int(octaves)):
        h0 = max(1, H // block)
        w0 = max(1, W // block)
        small = rng.uniform(0.0, 1.0, size=(h0, w0)).astype(dtype)
        up = _upsample_repeat(small, block, H, W)
        tex = tex + xp.asarray(amp, dtype=dtype) * up
        total_amp += amp
        amp *= 0.5
        block = max(1, block // 2)

    tex = tex / xp.asarray(total_amp, dtype=dtype)
    if blur_iters and blur_iters > 0:
        tex = _box_blur2d(tex, iters=int(blur_iters))
    tex = _normalize01(tex, dtype)
    return tex


def _to_nchw(tex2d, channels, rng, dtype, colorize=True):
    """
    tex2d: (H,W) in [0,1]
    returns img: (C,H,W) in [0,1]
    """
    H, W = tex2d.shape
    if channels == 1:
        return tex2d[None, :, :].astype(dtype)

    img = xp.zeros((3, H, W), dtype=dtype)

    if colorize:
        # random linear mix into RGB (keeps it “texture-ish”)
        a = rng.uniform(0.2, 1.0, size=(3,)).astype(dtype)
        b = rng.uniform(0.0, 0.4, size=(3,)).astype(dtype)
        for c in range(3):
            img[c] = xp.clip(a[c] * tex2d + b[c], 0.0, 1.0)
    else:
        for c in range(3):
            img[c] = tex2d
    return img.astype(dtype)


def _pad_1d(seq, length, pad_token):
    out = [pad_token] * length
    n = min(len(seq), length)
    out[:n] = seq[:n]
    return out, n


def _make_lengths(rng, n_samples, min_len, max_len):
    if min_len > max_len:
        raise ValueError("min_len must be <= max_len")
    return rng.randint(int(min_len), int(max_len) + 1, size=(n_samples,), dtype=xp.int64)


def _shift_for_next_token(seqs, pad_token):
    """
    seqs: (N, T) int64
    returns:
      X = seqs[:, :-1]
      Y = seqs[:, 1:]
    """
    return seqs[:, :-1], seqs[:, 1:]


def _ensure_nchw_digits(digits, dtype):
    """
    Accepts digits as:
      (N, 28, 28) or (N, 1, 28, 28) or (N, C, H, W)
    Returns:
      digits: (N, 1, H, W) float in [0,1]
    """
    d = xp.asarray(digits)
    if d.ndim == 3:
        d = d[:, None, :, :]
    elif d.ndim == 4:
        pass
    else:
        raise ValueError("digits must be (N,H,W) or (N,C,H,W)")

    # force 1 channel
    if d.shape[1] != 1:
        d = d[:, :1, :, :]

    d = d.astype(dtype)
    # normalize if looks like 0..255
    mx = float(xp.max(d))
    if mx > 1.5:
        d = d / xp.asarray(255.0, dtype=dtype)
    d = xp.clip(d, 0.0, 1.0)
    return d


def _resolve_wall_bounce(x, y, vx, vy, r, W, H):
    # x,y,vx,vy are python floats; r float
    # left/right
    if x - r < 0:
        x = r
        vx = -vx
    elif x + r > (W - 1):
        x = (W - 1) - r
        vx = -vx
    # top/bottom
    if y - r < 0:
        y = r
        vy = -vy
    elif y + r > (H - 1):
        y = (H - 1) - r
        vy = -vy
    return x, y, vx, vy


def _resolve_pair_collision(pi, pj):
    """
    Elastic collision for equal masses in 2D.
    pi/pj are dicts with x,y,vx,vy,r.
    Returns collision_happened (bool).
    """
    dx = pj["x"] - pi["x"]
    dy = pj["y"] - pi["y"]
    rr = pi["r"] + pj["r"]
    dist2 = dx * dx + dy * dy
    if dist2 <= 1e-12:
        return False
    if dist2 >= rr * rr:
        return False

    dist = math.sqrt(dist2)
    nx = dx / dist
    ny = dy / dist

    # relative velocity along normal
    rvx = pi["vx"] - pj["vx"]
    rvy = pi["vy"] - pj["vy"]
    vn = rvx * nx + rvy * ny

    # if moving apart, skip (prevents jitter)
    if vn > 0:
        # still push them apart a bit to avoid sticking
        overlap = rr - dist
        push = overlap * 0.5
        pi["x"] -= nx * push
        pi["y"] -= ny * push
        pj["x"] += nx * push
        pj["y"] += ny * push
        return True

    # impulse for equal masses: swap normal components
    pi["vx"] -= vn * nx
    pi["vy"] -= vn * ny
    pj["vx"] += vn * nx
    pj["vy"] += vn * ny

    # positional correction to un-overlap
    overlap = rr - dist
    push = overlap * 0.5
    pi["x"] -= nx * push
    pi["y"] -= ny * push
    pj["x"] += nx * push
    pj["y"] += ny * push

    return True


def _square_mask(yy, xx, cy, cx, r):
    # axis-aligned square inscribed in a circle of radius r (so half-side ~ r / sqrt(2))
    half = r / math.sqrt(2.0)
    return (xp.abs(yy - cy) <= half) & (xp.abs(xx - cx) <= half)