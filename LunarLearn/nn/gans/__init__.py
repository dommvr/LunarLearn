from .ganloss import GANLoss
from .utils import vanilla
from .utils import lsgan
from .utils import hinge
from .utils import wasserstein
from .utils import sample_noise
from .utils import gradient_penalty

__all__ = [
    "GANLoss",
    "vanilla",
    "lsgan",
    "hinge",
    "wasserstein",
    "sample_noise",
    "gradient_penalty"
]