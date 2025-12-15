import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _make_grid)

xp = backend.xp
DTYPE = backend.DTYPE


def make_gaussian_blobs_segmentation(
    n_samples=200,
    image_size=128,
    n_blobs=6,
    sigma_range=(6.0, 20.0),   # in pixels
    channels=1,                # NCHW
    noise_std=0.02,
    return_image=True,
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Gaussian blobs segmentation:
      pick n_blobs centers, compute Gaussian score maps, assign each pixel to argmax blob.

    Output per sample:
      {"image": (C,H,W), "mask": (H,W)}  mask in [0..n_blobs-1]
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if n_blobs < 2:
        raise ValueError("n_blobs must be >= 2")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    samples = []

    for _ in range(n_samples):
        cy = rng.uniform(0.0, H - 1.0, size=(n_blobs,)).astype(dtype)
        cx = rng.uniform(0.0, W - 1.0, size=(n_blobs,)).astype(dtype)
        sig = rng.uniform(float(sigma_range[0]), float(sigma_range[1]), size=(n_blobs,)).astype(dtype)

        dy = yy[None, :, :] - cy[:, None, None]
        dx = xx[None, :, :] - cx[:, None, None]
        dist2 = dy * dy + dx * dx

        # scores: exp(-dist2/(2*sigma^2))
        denom = 2.0 * (sig[:, None, None] ** 2) + xp.asarray(1e-12, dtype=dtype)
        scores = xp.exp(-dist2 / denom)

        mask = xp.argmax(scores, axis=0).astype(xp.int64)

        out = {"mask": mask}

        if return_image:
            img = xp.zeros((channels, H, W), dtype=dtype)
            # Render using normalized max score to create smooth-ish intensity
            smax = xp.max(scores, axis=0)
            smax = smax / (xp.max(smax) + xp.asarray(1e-12, dtype=dtype))

            if channels == 1:
                img[0] = xp.clip(0.15 + 0.85 * smax, 0.0, 1.0)
            else:
                cols = rng.uniform(0.2, 1.0, size=(n_blobs, 3)).astype(dtype)
                for c in range(3):
                    img[c] = xp.clip(0.15 + 0.85 * smax * cols[:, c][mask], 0.0, 1.0)

            if noise_std and noise_std > 0:
                img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

            out["image"] = xp.clip(img, 0.0, 1.0).astype(dtype)

        samples.append(out)

    if shuffle:
        perm = _get_rng(random_state).permutation(len(samples))
        samples = [samples[i] for i in perm.tolist()]

    return samples