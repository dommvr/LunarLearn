import LunarLearn.core.backend.backend as backend
from LunarLearn.data.augmentation.utils import (_rand,
                                                _ensure_ct,
                                                _as_xp,
                                                _restore_shape,
                                                _interp1d_ct,
                                                _smooth1d,
                                                _permutation)

xp = backend.xp
DTYPE = backend.DTYPE


# ----------------------------
# Augmentations
# ----------------------------

class Jitter:
    """
    Add Gaussian noise: x += N(0, std^2)
    std can be scalar or per-channel.
    """
    def __init__(self, std=0.01, p=0.5, dtype=None):
        self.std = std
        self.p = float(p)
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(_as_xp(x, self.dtype))
        C, T = X.shape
        std = xp.asarray(self.std, dtype=X.dtype)
        if std.ndim == 0:
            std = xp.full((C, 1), float(std), dtype=X.dtype)
        elif std.ndim == 1:
            std = std.reshape(C, 1)
        noise = xp.random.normal(0.0, 1.0, size=(C, T)).astype(X.dtype) * std
        Y = X + noise
        return _restore_shape(Y, orig_ndim)


class Scaling:
    """
    Multiply by random factor ~ N(1, sigma^2) (global or per-channel).
    """
    def __init__(self, sigma=0.05, per_channel=False, p=0.5, dtype=None):
        self.sigma = float(sigma)
        self.per_channel = bool(per_channel)
        self.p = float(p)
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(_as_xp(x, self.dtype))
        C, T = X.shape
        if self.per_channel:
            m = xp.random.normal(1.0, self.sigma, size=(C, 1)).astype(X.dtype)
        else:
            m = xp.asarray(float(xp.random.normal(1.0, self.sigma)), dtype=X.dtype)
        Y = X * m
        return _restore_shape(Y, orig_ndim)


class TimeShift:
    """
    Roll along time axis by random shift in [-max_shift, max_shift].
    """
    def __init__(self, max_shift=10, p=0.5):
        self.max_shift = int(max_shift)
        self.p = float(p)

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(xp.asarray(x))
        T = int(X.shape[1])
        s = int(xp.random.randint(-self.max_shift, self.max_shift + 1))
        s = int(s % T) if T > 0 else 0
        Y = xp.roll(X, shift=s, axis=1)
        return _restore_shape(Y, orig_ndim)


class WindowSlicing:
    """
    Take a random window and resample back to original length.
    Equivalent to cropping then resize in time.
    """
    def __init__(self, min_frac=0.7, p=0.5, mode="linear", dtype=None):
        self.min_frac = float(min_frac)
        self.p = float(p)
        self.mode = mode
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(_as_xp(x, self.dtype))
        C, T = X.shape
        if T < 2:
            return _restore_shape(X, orig_ndim)

        frac = float(xp.random.uniform(self.min_frac, 1.0))
        L = max(2, int(round(frac * T)))
        start = int(xp.random.randint(0, T - L + 1))
        win = X[:, start:start+L]

        # resample back to T
        t_src = xp.linspace(0, L - 1, T, dtype=X.dtype)
        Y = _interp1d_ct(win, t_src, mode=self.mode)
        return _restore_shape(Y, orig_ndim)


class Permutation:
    """
    Split into n_segments and shuffle them.
    """
    def __init__(self, n_segments=4, p=0.5):
        self.n = int(n_segments)
        self.p = float(p)

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(xp.asarray(x))
        C, T = X.shape
        n = max(2, min(self.n, T))
        # segment boundaries
        cuts = xp.linspace(0, T, n + 1, dtype=xp.int64)
        segs = []
        for i in range(n):
            a = int(cuts[i])
            b = int(cuts[i+1])
            segs.append(X[:, a:b])
        order = _permutation(n)
        Y = xp.concatenate([segs[i] for i in order], axis=1)
        return _restore_shape(Y, orig_ndim)


class TimeMask:
    """
    Mask a contiguous time window (set to value).
    """
    def __init__(self, max_frac=0.2, value=0.0, p=0.5, dtype=None):
        self.max_frac = float(max_frac)
        self.value = value
        self.p = float(p)
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(_as_xp(x, self.dtype)).copy()
        C, T = X.shape
        if T < 2:
            return _restore_shape(X, orig_ndim)
        L = max(1, int(round(float(xp.random.uniform(0.0, self.max_frac)) * T)))
        start = int(xp.random.randint(0, max(1, T - L + 1)))
        X[:, start:start+L] = xp.asarray(self.value, dtype=X.dtype)
        return _restore_shape(X, orig_ndim)


class TimeWarp:
    """
    Time warping via a smooth random monotonic warp function.
    Implementation:
      - sample random noise
      - smooth it
      - exponentiate to keep positive "speed"
      - integrate to get increasing mapping
      - normalize to [0, T-1]
      - resample
    """
    def __init__(self, sigma=0.2, smooth=9, p=0.3, mode="linear", dtype=None):
        self.sigma = float(sigma)
        self.smooth = int(smooth)
        self.p = float(p)
        self.mode = mode
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(_as_xp(x, self.dtype))
        C, T = X.shape
        if T < 3:
            return _restore_shape(X, orig_ndim)

        noise = xp.random.normal(0.0, self.sigma, size=(T,)).astype(X.dtype)
        speed = xp.exp(_smooth1d(noise, k=self.smooth))  # positive
        cum = xp.cumsum(speed)
        # map output indices -> input indices
        t_src = (cum / (cum[-1] + 1e-12)) * (T - 1)
        Y = _interp1d_ct(X, t_src, mode=self.mode)
        return _restore_shape(Y, orig_ndim)


class MagnitudeWarp:
    """
    Smooth varying scale over time: x(t) *= s(t)
    """
    def __init__(self, sigma=0.2, smooth=9, per_channel=False, p=0.3, dtype=None):
        self.sigma = float(sigma)
        self.smooth = int(smooth)
        self.per_channel = bool(per_channel)
        self.p = float(p)
        self.dtype = dtype

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(_as_xp(x, self.dtype))
        C, T = X.shape
        if T < 2:
            return _restore_shape(X, orig_ndim)

        if self.per_channel:
            scales = []
            for _ in range(C):
                n = xp.random.normal(0.0, self.sigma, size=(T,)).astype(X.dtype)
                s = xp.exp(_smooth1d(n, k=self.smooth))
                scales.append(s[None, :])
            S = xp.concatenate(scales, axis=0)  # (C,T)
        else:
            n = xp.random.normal(0.0, self.sigma, size=(T,)).astype(X.dtype)
            s = xp.exp(_smooth1d(n, k=self.smooth))
            S = s[None, :]

        Y = X * S
        return _restore_shape(Y, orig_ndim)


class FrequencyDropout:
    """
    FFT-based band-stop / dropout.
    - Keep it optional-ish: requires xp.fft.
    - Applies a random contiguous frequency band mask.
    """
    def __init__(self, drop_frac=0.1, p=0.2):
        self.drop_frac = float(drop_frac)
        self.p = float(p)

    def __call__(self, x):
        if _rand() > self.p:
            return x
        orig_ndim = xp.asarray(x).ndim
        X = _ensure_ct(xp.asarray(x))
        C, T = X.shape
        if T < 4:
            return _restore_shape(X, orig_ndim)

        # rfft over time
        F = xp.fft.rfft(X, axis=1)
        K = int(F.shape[1])
        band = max(1, int(round(self.drop_frac * K)))
        start = int(xp.random.randint(0, max(1, K - band)))
        mask = xp.ones((K,), dtype=F.dtype)
        mask[start:start+band] = 0
        F2 = F * mask[None, :]
        Y = xp.fft.irfft(F2, n=T, axis=1).astype(X.dtype)
        return _restore_shape(Y, orig_ndim)


# ----------------------------
# Batch-level MixUp (sequence-level)
# ----------------------------

def timeseries_mixup_batch(X, y, alpha=0.2):
    """
    X: (N,C,T) or (N,T)
    y: (N,) or (N,K)
    """
    if alpha <= 0:
        return X, y

    X = xp.asarray(X)
    y = xp.asarray(y)
    N = int(X.shape[0])

    lam = float(xp.random.beta(alpha, alpha))
    perm = xp.random.permutation(N)

    X2 = lam * X + (1.0 - lam) * X[perm]
    if y.ndim == 1:
        y2 = lam * y.astype(DTYPE) + (1.0 - lam) * y[perm].astype(DTYPE)
    else:
        y2 = lam * y.astype(DTYPE) + (1.0 - lam) * y[perm].astype(DTYPE)
    return X2, y2