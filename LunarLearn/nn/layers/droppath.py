import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor

xp = backend.xp

class DropPath(BaseLayer):
    def __init__(self, keep_prob: float = 0.9):
        super().__init__(trainable=False)

        if not (0.0 < keep_prob <= 1.0):
            raise ValueError("keep_prob must be in (0, 1].")

        self.keep_prob = keep_prob

    def initialize(self, input_shape):
        self.output_shape = input_shape

    def forward(self, A_prev: Tensor) -> Tensor:
        if not self.training or self.keep_prob == 1.0:
            return A_prev

        shape = (A_prev.shape[0],) + (1,) * (A_prev.ndim - 1)
        mask = (xp.random.rand(*shape) < self.keep_prob).astype(A_prev.dtype)
        mask = Tensor(mask, requires_grad=False, dtype=A_prev.dtype)
        scale = Tensor(1.0 / self.keep_prob, requires_grad=False, dtype=A_prev.dtype)

        return (A_prev * mask) * scale