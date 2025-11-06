import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class LambdaLayer(BaseLayer):
    def __init__(self, fn):
        super().__init__(trainable=False)

        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        out = self.fn(x)

        return out