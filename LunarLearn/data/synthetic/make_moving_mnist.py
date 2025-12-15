import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _ensure_nchw_digits)

xp = backend.xp
DTYPE = backend.DTYPE


def make_moving_mnist(
    digits,                      # array of digit images (N,1,28,28) or (N,28,28)
    n_sequences=1000,
    seq_len=20,
    frame_size=64,
    n_digits=2,                  # digits per frame
    velocity_range=(1, 3),       # pixels per step (integer)
    bounce=True,
    compose="max",               # "max" or "sum"
    noise_std=0.0,
    shuffle=True,
    random_state=None,
    dtype=None,
    return_next_frame_pairs=False,
):
    """
    Moving MNIST generator.

    Returns:
      video: (N, T, C, H, W) float [0,1]
      optionally X, Y for next-frame prediction:
        X: (N, T-1, C, H, W)
        Y: (N, T-1, C, H, W)
    """
    if dtype is None:
        dtype = DTYPE

    rng = _get_rng(random_state)

    digits = _ensure_nchw_digits(digits, dtype=dtype)
    n_src = int(digits.shape[0])
    C = 1
    H = W = int(frame_size)
    T = int(seq_len)

    if n_digits < 1:
        raise ValueError("n_digits must be >= 1")
    if H < digits.shape[2] or W < digits.shape[3]:
        raise ValueError("frame_size must be >= digit size")
    if compose not in ("max", "sum"):
        raise ValueError('compose must be "max" or "sum"')
    if velocity_range[0] < 0 or velocity_range[1] < velocity_range[0]:
        raise ValueError("invalid velocity_range")

    dh = int(digits.shape[2])
    dw = int(digits.shape[3])

    video = xp.zeros((n_sequences, T, C, H, W), dtype=dtype)

    for n in range(n_sequences):
        # pick digit images for this sequence
        idx = rng.randint(0, n_src, size=(n_digits,), dtype=xp.int64)
        sprites = digits[idx]  # (n_digits,1,dh,dw)

        # initial positions (top-left corners)
        xs = rng.randint(0, W - dw + 1, size=(n_digits,), dtype=xp.int64).astype(xp.int64)
        ys = rng.randint(0, H - dh + 1, size=(n_digits,), dtype=xp.int64).astype(xp.int64)

        # velocities (non-zero)
        vmin, vmax = int(velocity_range[0]), int(velocity_range[1])
        if vmax == 0:
            v_choices = [0]
        else:
            v_choices = list(range(-vmax, -vmin + 1)) + list(range(vmin, vmax + 1))
            if vmin == 0:
                v_choices = list(range(-vmax, 0)) + list(range(1, vmax + 1))

        vx = xp.asarray([v_choices[int(rng.randint(0, len(v_choices)))] for _ in range(n_digits)], dtype=xp.int64)
        vy = xp.asarray([v_choices[int(rng.randint(0, len(v_choices)))] for _ in range(n_digits)], dtype=xp.int64)

        for t in range(T):
            frame = xp.zeros((C, H, W), dtype=dtype)

            for k in range(n_digits):
                x = int(xs[k])
                y = int(ys[k])

                patch = frame[:, y:y+dh, x:x+dw]
                sprite = sprites[k]

                if compose == "max":
                    patch = xp.maximum(patch, sprite)
                else:
                    patch = patch + sprite

                frame[:, y:y+dh, x:x+dw] = patch

            if compose == "sum":
                frame = xp.clip(frame, 0.0, 1.0)

            if noise_std and noise_std > 0:
                frame = frame + rng.normal(0.0, float(noise_std), size=frame.shape).astype(dtype)
                frame = xp.clip(frame, 0.0, 1.0)

            video[n, t] = frame

            # update positions
            xs = xs + vx
            ys = ys + vy

            if bounce:
                for k in range(n_digits):
                    # x bounce
                    if xs[k] < 0:
                        xs[k] = 0
                        vx[k] = -vx[k]
                    elif xs[k] > (W - dw):
                        xs[k] = W - dw
                        vx[k] = -vx[k]
                    # y bounce
                    if ys[k] < 0:
                        ys[k] = 0
                        vy[k] = -vy[k]
                    elif ys[k] > (H - dh):
                        ys[k] = H - dh
                        vy[k] = -vy[k]
            else:
                # wrap
                xs = xs % xp.asarray(W - dw + 1, dtype=xp.int64)
                ys = ys % xp.asarray(H - dh + 1, dtype=xp.int64)

    if shuffle:
        perm = rng.permutation(n_sequences)
        video = video[perm]

    video = video.astype(dtype)

    if return_next_frame_pairs:
        X = video[:, :-1]
        Y = video[:, 1:]
        return video, X, Y

    return video