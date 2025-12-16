import LunarLearn.core.backend.backend as backend

xp = backend.xp
DTYPE = backend.xp


def _as_float_img(x, dtype=None):
    if dtype is None:
        dtype = x.dtype if hasattr(x, "dtype") else DTYPE
    x = xp.asarray(x, dtype=dtype)
    return x

def _clamp01(x):
    return xp.clip(x, 0.0, 1.0)

def _grid_sample_chw(img, grid_y, grid_x, mode="bilinear", padding="zeros"):
    """
    img: (C,H,W)
    grid_y, grid_x: (Hout, Wout) float coords in source pixel space
    """
    C, H, W = img.shape
    Hout, Wout = grid_y.shape

    if mode == "nearest":
        yi = xp.rint(grid_y).astype(xp.int64)
        xi = xp.rint(grid_x).astype(xp.int64)
        if padding == "border":
            yi = xp.clip(yi, 0, H - 1)
            xi = xp.clip(xi, 0, W - 1)
            out = img[:, yi, xi]
            return out
        else:
            valid = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
            yi2 = xp.clip(yi, 0, H - 1)
            xi2 = xp.clip(xi, 0, W - 1)
            out = img[:, yi2, xi2]
            out = out * valid[None, :, :]
            return out

    # bilinear
    y0 = xp.floor(grid_y).astype(xp.int64)
    x0 = xp.floor(grid_x).astype(xp.int64)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = grid_y - y0.astype(grid_y.dtype)
    wx = grid_x - x0.astype(grid_x.dtype)

    def sample(yy, xx):
        if padding == "border":
            yy = xp.clip(yy, 0, H - 1)
            xx = xp.clip(xx, 0, W - 1)
            return img[:, yy, xx], None
        else:
            valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            yy2 = xp.clip(yy, 0, H - 1)
            xx2 = xp.clip(xx, 0, W - 1)
            v = img[:, yy2, xx2]
            return v, valid

    Ia, va = sample(y0, x0)
    Ib, vb = sample(y0, x1)
    Ic, vc = sample(y1, x0)
    Id, vd = sample(y1, x1)

    wa = (1 - wy) * (1 - wx)
    wb = (1 - wy) * wx
    wc = wy * (1 - wx)
    wd = wy * wx

    out = Ia * wa[None, :, :] + Ib * wb[None, :, :] + Ic * wc[None, :, :] + Id * wd[None, :, :]

    if padding != "border":
        va = va.astype(out.dtype); vb = vb.astype(out.dtype); vc = vc.astype(out.dtype); vd = vd.astype(out.dtype)
        out = out * (va[None]*wa[None] + vb[None]*wb[None] + vc[None]*wc[None] + vd[None]*wd[None] > 0).astype(out.dtype)

    return out


def resize_chw(img, out_h, out_w, mode="bilinear"):
    img = _as_float_img(img)
    C, H, W = img.shape
    yy = xp.linspace(0, H - 1, int(out_h), dtype=img.dtype)
    xx = xp.linspace(0, W - 1, int(out_w), dtype=img.dtype)
    grid_y, grid_x = xp.meshgrid(yy, xx, indexing="ij")
    return _grid_sample_chw(img, grid_y, grid_x, mode=mode, padding="border")


def center_crop_chw(img, crop_h, crop_w):
    img = _as_float_img(img)
    C, H, W = img.shape
    ch, cw = int(crop_h), int(crop_w)
    y0 = max(0, (H - ch) // 2)
    x0 = max(0, (W - cw) // 2)
    return img[:, y0:y0+ch, x0:x0+cw]


def pad_chw(img, pad, value=0.0):
    img = _as_float_img(img)
    C, H, W = img.shape
    if isinstance(pad, int):
        pt = pb = pl = pr = pad
    else:
        pt, pb, pl, pr = pad
    out = xp.full((C, H+pt+pb, W+pl+pr), value, dtype=img.dtype)
    out[:, pt:pt+H, pl:pl+W] = img
    return out


def affine_chw(img, M, out_h=None, out_w=None, mode="bilinear", padding="zeros"):
    """
    M: 2x3 mapping output -> input (pixel space)
    """
    img = _as_float_img(img)
    C, H, W = img.shape
    if out_h is None: out_h = H
    if out_w is None: out_w = W

    ys = xp.arange(int(out_h), dtype=img.dtype)
    xs = xp.arange(int(out_w), dtype=img.dtype)
    yy, xx = xp.meshgrid(ys, xs, indexing="ij")

    x_in = M[0,0]*xx + M[0,1]*yy + M[0,2]
    y_in = M[1,0]*xx + M[1,1]*yy + M[1,2]
    return _grid_sample_chw(img, y_in, x_in, mode=mode, padding=padding)


def perspective_chw(img, Hm, out_h=None, out_w=None, mode="bilinear", padding="zeros"):
    """
    Hm: 3x3 homography mapping output -> input (pixel space)
    """
    img = _as_float_img(img)
    C, H, W = img.shape
    if out_h is None: out_h = H
    if out_w is None: out_w = W

    ys = xp.arange(int(out_h), dtype=img.dtype)
    xs = xp.arange(int(out_w), dtype=img.dtype)
    yy, xx = xp.meshgrid(ys, xs, indexing="ij")

    denom = Hm[2,0]*xx + Hm[2,1]*yy + Hm[2,2]
    x_in = (Hm[0,0]*xx + Hm[0,1]*yy + Hm[0,2]) / denom
    y_in = (Hm[1,0]*xx + Hm[1,1]*yy + Hm[1,2]) / denom
    return _grid_sample_chw(img, y_in, x_in, mode=mode, padding=padding)


# homography helper (DLT)
def _homography_from_4pts(dst_pts, src_pts, dtype):
    # dst_pts: (x,y) output, src_pts: (x,y) input
    A = []
    b = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v])
        b.append(u)
        b.append(v)
    A = xp.asarray(A, dtype=dtype)
    b = xp.asarray(b, dtype=dtype).reshape(-1, 1)

    # solve A h = b
    # h: (8,1), last element is 1
    h = xp.linalg.solve(A, b).reshape(-1)
    Hm = xp.asarray([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]], dtype=dtype)
    return Hm


def _gaussian_kernel1d(sigma, radius, dtype):
    xs = xp.arange(-radius, radius + 1, dtype=dtype)
    k = xp.exp(-(xs**2) / (2.0 * sigma * sigma))
    k = k / xp.sum(k)
    return k


def _conv1d_h(img, k):
    C, H, W = img.shape
    r = int((k.shape[0] - 1) // 2)
    pad = pad_chw(img, (0, 0, r, r), value=0.0)
    out = xp.zeros_like(img)
    for i in range(W):
        window = pad[:, :, i:i+2*r+1]  # (C,H,K)
        out[:, :, i] = xp.sum(window * k[None, None, :], axis=2)
    return out


def _conv1d_v(img, k):
    C, H, W = img.shape
    r = int((k.shape[0] - 1) // 2)
    pad = pad_chw(img, (r, r, 0, 0), value=0.0)
    out = xp.zeros_like(img)
    for i in range(H):
        window = pad[:, i:i+2*r+1, :]  # (C,K,W)
        out[:, i, :] = xp.sum(window * k[None, :, None], axis=1)
    return out
