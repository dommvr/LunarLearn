import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE

class WeightNorm:
    def __init__(self, param: Parameter, dim=0, epsilon=1e-8):
        self.dim = dim
        self.epsilon = epsilon

        self.v = Parameter(param.master.copy(), requires_grad=True)
        g = ops.norm(self.v, axis=dim, keepdims=True)
        self.g = Parameter(g, requires_grad=True)

        param.normalization = self

    def __call__(self, v: Tensor) -> Tensor:
        g = self.g.to_compute()
        v_norm = ops.norm(v, axis=self.dim, keepdims=True)
        v_hat = v / (v_norm + self.epsilon)
        out = g * v_hat
        return out
    
    def parameters(self):
        return [self.v, self.g]
    
    def named_parameters(self, prefix: str = ""):
        return [
            (f"{prefix}.v", self.v),
            (f"{prefix}.g", self.g)
        ]