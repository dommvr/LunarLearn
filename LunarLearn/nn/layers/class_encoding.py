import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp

class ClassEncoding(BaseLayer):
    def __init__(self, distillation=False):
        super().__init__(trainable=True)
        self.distillation = distillation
        self.cls_token = None
        self.dist_token = None

    def _initialize(self, input_shape):
        n_patches, d_model = input_shape
        cls_token = xp.random.randn(1, 1, d_model)
        self.cls_token = Parameter(cls_token, requires_grad=True)
        if self.distillation:
            dist_token = xp.random.randn(1, 1, d_model)
            self.dist_token = Parameter(dist_token, requires_grad=True)
        self.output_shape = (n_patches + 1 + (1 if self.distillation else 0), d_model)

    def forward(self, x: Tensor) -> Tensor:
        if self.cls_token is None:
            self._initialize(x.shape[1:])
        batch = x.shape[0]
        cls_token = self.cls_token.to_compute()
        tokens = cls_token.expand(batch, -1, -1)
        if self.dist_token is not None:
            dist_token = self.dist_token.to_compute()
            dist_tokens = dist_token.expand(batch, -1, -1)
            tokens = ops.concatenate([tokens, dist_tokens], axis=1)
        x = ops.concatenate([tokens, x], axis=1)
        return x