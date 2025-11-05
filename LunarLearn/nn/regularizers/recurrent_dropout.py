import LunarLearn.backend as backend
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class RecurrentDropout:
    def __init__(self, a: Tensor, recurrent_keep_prob: float, training: bool = True):
        self.training = training

        if training:
            self.mask, self.scale = self.initialize(a, recurrent_keep_prob)
        else:
            self.mask, self.scale = None, None

    def __call__(self, a: Tensor) -> Tensor:
        return self.drop(a)

    def initialize(self, shape, recurrent_keep_prob: float):
        # Generate mask
        dtype = C_DTYPE if MIXED_PRECISION else DTYPE
        mask = (xp.random.rand(*shape) < recurrent_keep_prob).astype(dtype)
        mask = Tensor(mask, requires_grad=False, dtype=dtype)

        # Scaling factor
        scale = Tensor(1.0 / recurrent_keep_prob, requires_grad=False, dtype=dtype)

        return mask, scale
    
    def drop(self, a: Tensor) -> Tensor:
        if self.mask is not None:
            a =  (a * self.mask) * self.scale
        
        return a