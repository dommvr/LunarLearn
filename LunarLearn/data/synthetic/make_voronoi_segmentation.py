import LunarLearn.core.backend.backend as backend
from LunarLearn.data.synthetic.utils import (_get_rng,
                                             _make_grid,
                                             _lowfreq_texture)

xp = backend.xp
DTYPE = backend.DTYPE


def make_voronoi_segmentation(
    n_samples=200,
    image_size=128,
    n_regions=8,
    channels=1,               # NCHW
    background="none",        # "none" | "lowfreq"
    noise_std=0.02,
    return_image=True,        # if False, return only mask in dict
    shuffle=True,
    random_state=None,
    dtype=None,
):
    """
    Voronoi cell segmentation.
    Output per sample:
      {"image": (C,H,W), "mask": (H,W)}  where mask in [0..n_regions-1]
    """
    if dtype is None:
        dtype = DTYPE
    H = W = int(image_size)
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    if n_regions < 2:
        raise ValueError("n_regions must be >= 2")

    rng = _get_rng(random_state)
    yy, xx = _make_grid(H, W, dtype=dtype)

    samples = []

    for _ in range(n_samples):
        # random sites in pixel coords
        sy = rng.uniform(0.0, H - 1.0, size=(n_regions,)).astype(dtype)
        sx = rng.uniform(0.0, W - 1.0, size=(n_regions,)).astype(dtype)

        # squared distances to each site: (n_regions,H,W)
        # broadcast: (n_regions,1,1) vs (H,W)
        dy = yy[None, :, :] - sy[:, None, None]
        dx = xx[None, :, :] - sx[:, None, None]
        dist2 = dy * dy + dx * dx

        mask = xp.argmin(dist2, axis=0).astype(xp.int64)  # (H,W)

        out = {"mask": mask}

        if return_image:
            img = xp.zeros((channels, H, W), dtype=dtype)

            if background == "lowfreq":
                tex = _lowfreq_texture(rng, H, W, dtype=dtype, block=max(2, H // 12))
                img += (xp.asarray(0.15, dtype=dtype) * tex)[None, :, :]

            # render regions as different intensities (or simple RGB)
            if channels == 1:
                vals = rng.uniform(0.2, 1.0, size=(n_regions,)).astype(dtype)
                img[0] = vals[mask]
            else:
                cols = rng.uniform(0.2, 1.0, size=(n_regions, 3)).astype(dtype)
                for c in range(3):
                    img[c] = cols[:, c][mask]

            if noise_std and noise_std > 0:
                img = img + rng.normal(0.0, float(noise_std), size=img.shape).astype(dtype)

            out["image"] = xp.clip(img, 0.0, 1.0).astype(dtype)

        samples.append(out)

    if shuffle:
        perm = _get_rng(random_state).permutation(len(samples))
        samples = [samples[i] for i in perm.tolist()]

    return samples