import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp

class ClassEncoding(BaseLayer):
    def __init__(self):
        super().__init__(trainable=True)
        self.P = None

    def _initialize(self, input_shape):
        n_patches, d_model = input_shape
        P = xp.random.randn(1, 1, d_model)
        self.P = Parameter(P, requires_grad=True)
        self.output_shape = (n_patches + 1, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if self.P is None:
            self._initialize(x.shape[1:])
        batch = x.shape[0]
        P = self.P.to_compute()
        cls_tokens = P.expand(batch, -1, -1)
        x = ops.concatenate([cls_tokens, x], axis=1)
        return x