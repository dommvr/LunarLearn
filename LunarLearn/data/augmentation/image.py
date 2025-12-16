import LunarLearn.core.backend.backend as backend
from LunarLearn.data.augmentation.utils import (_as_float_img,
                                                pad_chw,
                                                resize_chw,
                                                affine_chw,
                                                _homography_from_4pts,
                                                perspective_chw,
                                                center_crop_chw,
                                                _clamp01,
                                                _conv1d_h,
                                                _conv1d_v,
                                                _gaussian_kernel1d,
                                                _grid_sample_chw)
from LunarLearn.core import Tensor
import math

xp = backend.xp
DTYPE = backend.DTYPE


# --------------------------------------------
# Geometric augmentations
# --------------------------------------------

class RandomCropPadResize:
    def __init__(self, out_size, pad=4, p=1.0, mode="bilinear"):
        self.out_h, self.out_w = (out_size, out_size) if isinstance(out_size, int) else out_size
        self.pad = int(pad)
        self.p = float(p)
        self.mode = mode

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return resize_chw(img, self.out_h, self.out_w, mode=self.mode)

        x = pad_chw(img, self.pad, value=0.0)
        C, H, W = x.shape
        ch, cw = self.out_h, self.out_w
        if H == ch and W == cw:
            return x
        y0 = int(xp.random.randint(0, max(1, H - ch + 1)))
        x0 = int(xp.random.randint(0, max(1, W - cw + 1)))
        return x[:, y0:y0+ch, x0:x0+cw]


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, img):
        if float(xp.random.rand()) < self.p:
            return img[:, :, ::-1]
        return img


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = float(p)

    def __call__(self, img):
        if float(xp.random.rand()) < self.p:
            return img[:, ::-1, :]
        return img


class RandomRotation:
    def __init__(self, degrees=10.0, mode="bilinear", padding="zeros"):
        self.deg = float(degrees)
        self.mode = mode
        self.padding = padding

    def __call__(self, img):
        img = _as_float_img(img)
        C, H, W = img.shape
        a = float(xp.random.uniform(-self.deg, self.deg)) * math.pi / 180.0
        ca, sa = math.cos(a), math.sin(a)

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # output -> input mapping
        M = xp.asarray([
            [ ca,  sa, cx - ca*cx - sa*cy],
            [-sa,  ca, cy + sa*cx - ca*cy],
        ], dtype=img.dtype)

        return affine_chw(img, M, mode=self.mode, padding=self.padding)


class RandomTranslate:
    def __init__(self, max_shift=4, mode="bilinear", padding="zeros"):
        self.max_shift = float(max_shift)
        self.mode = mode
        self.padding = padding

    def __call__(self, img):
        img = _as_float_img(img)
        dx = float(xp.random.uniform(-self.max_shift, self.max_shift))
        dy = float(xp.random.uniform(-self.max_shift, self.max_shift))
        M = xp.asarray([[1.0, 0.0, dx],
                        [0.0, 1.0, dy]], dtype=img.dtype)
        return affine_chw(img, M, mode=self.mode, padding=self.padding)


class RandomShear:
    def __init__(self, shear_deg=10.0, mode="bilinear", padding="zeros"):
        self.shear = float(shear_deg) * math.pi / 180.0
        self.mode = mode
        self.padding = padding

    def __call__(self, img):
        img = _as_float_img(img)
        shx = float(xp.random.uniform(-self.shear, self.shear))
        shy = float(xp.random.uniform(-self.shear, self.shear))

        C, H, W = img.shape
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # shear matrix around center
        A = xp.asarray([[1.0, math.tan(shx)],
                        [math.tan(shy), 1.0]], dtype=img.dtype)

        # build output->input affine M
        # x_in = A00*(x-cx)+A01*(y-cy)+cx
        # y_in = A10*(x-cx)+A11*(y-cy)+cy
        M = xp.asarray([
            [A[0,0], A[0,1], cx - A[0,0]*cx - A[0,1]*cy],
            [A[1,0], A[1,1], cy - A[1,0]*cx - A[1,1]*cy],
        ], dtype=img.dtype)

        return affine_chw(img, M, mode=self.mode, padding=self.padding)


class RandomAffine:
    def __init__(self, degrees=10.0, translate=0.1, scale=(0.9, 1.1), shear_deg=10.0,
                 mode="bilinear", padding="zeros"):
        self.deg = float(degrees)
        self.translate = float(translate)
        self.scale = (float(scale[0]), float(scale[1]))
        self.shear = float(shear_deg)
        self.mode = mode
        self.padding = padding

    def __call__(self, img):
        img = _as_float_img(img)
        C, H, W = img.shape
        a = float(xp.random.uniform(-self.deg, self.deg)) * math.pi / 180.0
        ca, sa = math.cos(a), math.sin(a)

        sc = float(xp.random.uniform(self.scale[0], self.scale[1]))

        sh = float(xp.random.uniform(-self.shear, self.shear)) * math.pi / 180.0
        tsh = math.tan(sh)

        max_dx = self.translate * W
        max_dy = self.translate * H
        dx = float(xp.random.uniform(-max_dx, max_dx))
        dy = float(xp.random.uniform(-max_dy, max_dy))

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # rotation+scale
        R = xp.asarray([[ sc*ca,  sc*sa],
                        [-sc*sa,  sc*ca]], dtype=img.dtype)

        # shear in x (simple, effective)
        S = xp.asarray([[1.0, tsh],
                        [0.0, 1.0]], dtype=img.dtype)

        A = R @ S

        M = xp.asarray([
            [A[0,0], A[0,1], cx + dx - A[0,0]*cx - A[0,1]*cy],
            [A[1,0], A[1,1], cy + dy - A[1,0]*cx - A[1,1]*cy],
        ], dtype=img.dtype)

        return affine_chw(img, M, mode=self.mode, padding=self.padding)


class RandomZoom:
    def __init__(self, scale=(0.8, 1.2), mode="bilinear", padding="zeros"):
        self.scale = (float(scale[0]), float(scale[1]))
        self.mode = mode
        self.padding = padding

    def __call__(self, img):
        img = _as_float_img(img)
        C, H, W = img.shape
        sc = float(xp.random.uniform(self.scale[0], self.scale[1]))
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        M = xp.asarray([
            [sc, 0.0, cx - sc*cx],
            [0.0, sc, cy - sc*cy],
        ], dtype=img.dtype)
        return affine_chw(img, M, mode=self.mode, padding=self.padding)


class RandomPerspective:
    def __init__(self, distortion=0.2, p=0.5, mode="bilinear", padding="zeros"):
        self.d = float(distortion)
        self.p = float(p)
        self.mode = mode
        self.padding = padding

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        C, H, W = img.shape

        def jitter(x, y):
            return (x + float(xp.random.uniform(-self.d, self.d)) * W,
                    y + float(xp.random.uniform(-self.d, self.d)) * H)

        # destination corners (output)
        dst = [(0,0), (W-1,0), (W-1,H-1), (0,H-1)]
        # source corners (input) jittered
        src = [jitter(0,0), jitter(W-1,0), jitter(W-1,H-1), jitter(0,H-1)]

        # compute homography Hm that maps output->input: dst -> src
        Hm = _homography_from_4pts(dst, src, dtype=img.dtype)
        return perspective_chw(img, Hm, mode=self.mode, padding=self.padding)


class CenterCrop:
    def __init__(self, size):
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, img):
        return center_crop_chw(_as_float_img(img), self.h, self.w)


class FiveCrop:
    """
    Eval-time: returns list of 5 crops (TL, TR, BL, BR, Center).
    This intentionally returns a list, so your collate should keep lists.
    """
    def __init__(self, size):
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, img):
        img = _as_float_img(img)
        C, H, W = img.shape
        ch, cw = self.h, self.w
        tl = img[:, 0:ch, 0:cw]
        tr = img[:, 0:ch, W-cw:W]
        bl = img[:, H-ch:H, 0:cw]
        br = img[:, H-ch:H, W-cw:W]
        cc = center_crop_chw(img, ch, cw)
        return [tl, tr, bl, br, cc]


class TenCrop:
    """
    FiveCrop + horizontal flip of each crop (10 total).
    """
    def __init__(self, size):
        self.five = FiveCrop(size)

    def __call__(self, img):
        crops = self.five(img)
        flips = [c[:, :, ::-1] for c in crops]
        return crops + flips
    

# --------------------------------------------
# Photometric augmentations (assume [0,1])
# --------------------------------------------

class ColorJitter:
    """
    Brightness/contrast/saturation/hue jitter.
    For saturation/hue, assumes 3-channel.
    Hue is implemented by rotating UV in YUV-ish space (cheap & decent).
    """
    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=1.0):
        self.b = float(brightness)
        self.c = float(contrast)
        self.s = float(saturation)
        self.h = float(hue)
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img

        x = img

        # brightness
        if self.b > 0:
            delta = float(xp.random.uniform(-self.b, self.b))
            x = x + delta

        # contrast
        if self.c > 0:
            factor = float(xp.random.uniform(1 - self.c, 1 + self.c))
            mean = xp.mean(x, axis=(1,2), keepdims=True)
            x = (x - mean) * factor + mean

        # saturation and hue only for RGB
        if x.shape[0] == 3:
            # grayscale for saturation
            if self.s > 0:
                sf = float(xp.random.uniform(1 - self.s, 1 + self.s))
                gray = (0.2989*x[0] + 0.5870*x[1] + 0.1140*x[2])[None, :, :]
                x = gray + sf * (x - gray)

            # hue rotate in UV space (approx)
            if self.h > 0:
                ang = float(xp.random.uniform(-self.h, self.h)) * math.pi
                ca, sa = math.cos(ang), math.sin(ang)
                # YUV-ish
                Y = 0.299*x[0] + 0.587*x[1] + 0.114*x[2]
                U = -0.14713*x[0] - 0.28886*x[1] + 0.436*x[2]
                V = 0.615*x[0] - 0.51499*x[1] - 0.10001*x[2]
                U2 = ca*U - sa*V
                V2 = sa*U + ca*V
                R = Y + 1.13983*V2
                G = Y - 0.39465*U2 - 0.58060*V2
                B = Y + 2.03211*U2
                x = xp.stack([R, G, B], axis=0)

        return _clamp01(x)


class GammaCorrection:
    def __init__(self, gamma=(0.8, 1.2), p=0.5):
        self.g0, self.g1 = float(gamma[0]), float(gamma[1])
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        g = float(xp.random.uniform(self.g0, self.g1))
        return _clamp01(xp.power(_clamp01(img), g))


class RandomGrayscale:
    def __init__(self, p=0.2):
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if img.shape[0] != 3 or float(xp.random.rand()) > self.p:
            return img
        gray = (0.2989*img[0] + 0.5870*img[1] + 0.1140*img[2])[None, :, :]
        return xp.repeat(gray, 3, axis=0)


class Invert:
    def __init__(self, p=0.2):
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) < self.p:
            return 1.0 - img
        return img


class Solarize:
    def __init__(self, threshold=0.5, p=0.2):
        self.t = float(threshold)
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        return xp.where(img >= self.t, 1.0 - img, img)


class Posterize:
    def __init__(self, bits=4, p=0.2):
        self.bits = int(bits)
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        levels = float(2 ** self.bits - 1)
        return xp.round(img * levels) / levels


class HistogramEqualize:
    """
    Simple per-channel histogram equalization (not CLAHE).
    Works on [0,1] float images.
    """
    def __init__(self, bins=256, p=0.2):
        self.bins = int(bins)
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img

        C, H, W = img.shape
        out = xp.empty_like(img)
        for c in range(C):
            x = _clamp01(img[c]).ravel()
            hist, edges = xp.histogram(x, bins=self.bins, range=(0.0, 1.0))
            cdf = xp.cumsum(hist).astype(img.dtype)
            cdf = cdf / (cdf[-1] + 1e-12)
            # map values to bins
            idx = xp.clip((x * (self.bins - 1)).astype(xp.int64), 0, self.bins - 1)
            y = cdf[idx]
            out[c] = y.reshape(H, W)
        return out


# --------------------------------------------
# Noise / blur / compression-ish
# --------------------------------------------

class GaussianNoise:
    def __init__(self, std=0.05, p=0.5):
        self.std = float(std)
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        noise = xp.random.normal(0.0, self.std, size=img.shape).astype(img.dtype)
        return _clamp01(img + noise)


class SaltPepperNoise:
    def __init__(self, amount=0.02, salt_vs_pepper=0.5, p=0.5):
        self.amount = float(amount)
        self.svp = float(salt_vs_pepper)
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        mask = xp.random.rand(*img.shape[1:]) < self.amount
        salt = xp.random.rand(*img.shape[1:]) < self.svp
        out = img.copy()
        out[:, mask & salt] = 1.0
        out[:, mask & (~salt)] = 0.0
        return out
    

class GaussianBlur:
    def __init__(self, sigma=(0.5, 1.5), p=0.3):
        self.s0, self.s1 = float(sigma[0]), float(sigma[1])
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        s = float(xp.random.uniform(self.s0, self.s1))
        radius = max(1, int(3 * s))
        k = _gaussian_kernel1d(s, radius, img.dtype)
        x = _conv1d_h(img, k)
        x = _conv1d_v(x, k)
        return _clamp01(x)


class MotionBlur:
    def __init__(self, ksize=7, p=0.2):
        self.ksize = int(ksize)
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        k = self.ksize
        # random direction: horizontal or vertical (cheap)
        if int(xp.random.randint(0, 2)) == 0:
            ker = xp.ones((k,), dtype=img.dtype) / float(k)
            return _clamp01(_conv1d_h(img, ker))
        else:
            ker = xp.ones((k,), dtype=img.dtype) / float(k)
            return _clamp01(_conv1d_v(img, ker))


class FakeJPEGArtifacts:
    """
    Not real JPEG. Simulates blockiness + quantization.
    Surprisingly useful anyway.
    """
    def __init__(self, block=8, q=(8, 32), p=0.2):
        self.block = int(block)
        self.q0, self.q1 = int(q[0]), int(q[1])
        self.p = float(p)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        C, H, W = img.shape
        q = int(xp.random.randint(self.q0, self.q1 + 1))
        out = img.copy()
        b = self.block
        for y in range(0, H, b):
            for x in range(0, W, b):
                patch = out[:, y:y+b, x:x+b]
                # quantize coarsely
                patch = xp.round(patch * q) / q
                out[:, y:y+b, x:x+b] = patch
        return _clamp01(out)


# --------------------------------------------
# Occlusion / mixing
# --------------------------------------------

class RandomErasing:
    def __init__(self, p=0.25, area=(0.02, 0.2), aspect=(0.3, 3.3), value="random"):
        self.p = float(p)
        self.a0, self.a1 = float(area[0]), float(area[1])
        self.r0, self.r1 = float(aspect[0]), float(aspect[1])
        self.value = value

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        C, H, W = img.shape
        A = H * W
        target = float(xp.random.uniform(self.a0, self.a1)) * A
        ar = float(xp.random.uniform(self.r0, self.r1))
        h = int(round(math.sqrt(target * ar)))
        w = int(round(math.sqrt(target / ar)))
        h = max(1, min(h, H))
        w = max(1, min(w, W))
        y0 = int(xp.random.randint(0, H - h + 1))
        x0 = int(xp.random.randint(0, W - w + 1))
        out = img.copy()
        if self.value == "random":
            out[:, y0:y0+h, x0:x0+w] = xp.random.rand(C, h, w).astype(img.dtype)
        else:
            out[:, y0:y0+h, x0:x0+w] = float(self.value)
        return out


class Cutout(RandomErasing):
    pass


class GridMask:
    def __init__(self, p=0.2, d=(16, 32), ratio=0.5):
        self.p = float(p)
        self.d0, self.d1 = int(d[0]), int(d[1])
        self.ratio = float(ratio)

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        C, H, W = img.shape
        d = int(xp.random.randint(self.d0, self.d1 + 1))
        l = int(d * self.ratio)
        y0 = int(xp.random.randint(0, d))
        x0 = int(xp.random.randint(0, d))
        mask = xp.ones((H, W), dtype=img.dtype)
        for y in range(y0, H, d):
            mask[y:y+l, :] = 0
        for x in range(x0, W, d):
            mask[:, x:x+l] = 0
        return img * mask[None, :, :]


# MixUp / CutMix are batch-level transforms (they need two samples)
# They should be applied AFTER collation, in training step or a batch-transform hook.

def mixup_batch(X, Y, alpha=0.2):
    """
    X: (N, ...) xp array or Tensor
    Y: (N, K) one-hot or (N,) class indices (prefer one-hot)
    Returns mixed (X, Y).
    """
    if isinstance(X, Tensor):
        Xd = X.data
    else:
        Xd = X
    if isinstance(Y, Tensor):
        Yd = Y.data
    else:
        Yd = Y

    N = Xd.shape[0]
    lam = float(xp.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = xp.random.permutation(N)

    X2 = lam * Xd + (1 - lam) * Xd[perm]
    Y2 = lam * Yd + (1 - lam) * Yd[perm]

    if isinstance(X, Tensor):
        X2 = Tensor(X2, requires_grad=False)
    if isinstance(Y, Tensor):
        Y2 = Tensor(Y2, requires_grad=False)
    return X2, Y2


def cutmix_batch(X, Y, alpha=1.0):
    """
    CutMix for images.
    X: (N,C,H,W)
    Y: (N,K) preferred
    """
    if isinstance(X, Tensor):
        Xd = X.data
    else:
        Xd = X
    if isinstance(Y, Tensor):
        Yd = Y.data
    else:
        Yd = Y

    N, C, H, W = Xd.shape
    lam = float(xp.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = xp.random.permutation(N)

    # bbox area = (1-lam)
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = int(xp.random.randint(0, W))
    cy = int(xp.random.randint(0, H))

    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)

    X2 = Xd.copy()
    X2[:, :, y1:y2, x1:x2] = Xd[perm, :, y1:y2, x1:x2]

    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / float(W * H))
    Y2 = lam_adj * Yd + (1.0 - lam_adj) * Yd[perm]

    if isinstance(X, Tensor):
        X2 = Tensor(X2, requires_grad=False)
    if isinstance(Y, Tensor):
        Y2 = Tensor(Y2, requires_grad=False)
    return X2, Y2


# --------------------------------------------
# Deformation
# --------------------------------------------

class ElasticTransform:
    """
    Classic MNIST-ish elastic distortion using smoothed displacement fields.
    """
    def __init__(self, alpha=8.0, sigma=3.0, p=0.2, mode="bilinear", padding="border"):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.p = float(p)
        self.mode = mode
        self.padding = padding

    def __call__(self, img):
        img = _as_float_img(img)
        if float(xp.random.rand()) > self.p:
            return img
        C, H, W = img.shape

        # random displacement fields
        dx = xp.random.normal(0.0, 1.0, size=(H, W)).astype(img.dtype)
        dy = xp.random.normal(0.0, 1.0, size=(H, W)).astype(img.dtype)

        # smooth via gaussian blur (reuse our kernels)
        radius = max(1, int(3 * self.sigma))
        k = _gaussian_kernel1d(self.sigma, radius, img.dtype)

        dx = _conv1d_v(_conv1d_h(dx[None, :, :], k), k)[0]
        dy = _conv1d_v(_conv1d_h(dy[None, :, :], k), k)[0]

        dx = dx * self.alpha
        dy = dy * self.alpha

        ys = xp.arange(H, dtype=img.dtype)
        xs = xp.arange(W, dtype=img.dtype)
        yy, xx = xp.meshgrid(ys, xs, indexing="ij")

        grid_y = yy + dy
        grid_x = xx + dx
        return _grid_sample_chw(img, grid_y, grid_x, mode=self.mode, padding=self.padding)


class RandomDistortion:
    """
    Mild distortion: just elastic with smaller defaults.
    """
    def __init__(self, p=0.2):
        self.t = ElasticTransform(alpha=4.0, sigma=2.0, p=p)

    def __call__(self, img):
        return self.t(img)


# --------------------------------------------
# Normalization / standard
# --------------------------------------------

class Normalize:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = xp.asarray(mean, dtype=DTYPE).reshape(-1, 1, 1)
        self.std = xp.asarray(std, dtype=DTYPE).reshape(-1, 1, 1)
        self.eps = float(eps)

    def __call__(self, img):
        img = _as_float_img(img, dtype=DTYPE)
        return (img - self.mean) / (self.std + self.eps)


class RandomChannelDropout:
    def __init__(self, p=0.05, drop_value=0.0):
        self.p = float(p)
        self.drop_value = float(drop_value)

    def __call__(self, img):
        img = _as_float_img(img)
        C, H, W = img.shape
        if C <= 1 or float(xp.random.rand()) > self.p:
            return img
        c = int(xp.random.randint(0, C))
        out = img.copy()
        out[c] = self.drop_value
        return out