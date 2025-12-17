import LunarLearn.core.backend.backend as backend

xp = backend.xp


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomApply:
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = float(p)

    def __call__(self, sample):
        if float(xp.random.rand()) < self.p:
            return self.transform(sample)
        return sample


class RandomChoice:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample):
        i = int(xp.random.randint(0, len(self.transforms)))
        return self.transforms[i](sample)


class ApplyToXY:
    """
    Apply a transform to x only or (x,y) together for tuple samples.
    """
    def __init__(self, x_transform=None, xy_transform=None):
        self.x_t = x_transform
        self.xy_t = xy_transform

    def __call__(self, sample):
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("ApplyToXY expects (x,y) sample")
        x, y = sample[0], sample[1]
        if self.xy_t is not None:
            x, y = self.xy_t(x, y)
        elif self.x_t is not None:
            x = self.x_t(x)
        return (x, y) if len(sample) == 2 else (x, y, *sample[2:])


class ApplyToKey:
    """
    Apply an image transform to a specific dict key (e.g. "image").
    Leaves other fields unchanged.
    """
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, sample):
        if not isinstance(sample, dict):
            raise TypeError("ApplyToKey expects dict samples")
        out = dict(sample)
        out[self.key] = self.transform(out[self.key])
        return out