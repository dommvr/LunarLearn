import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Stateful
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE

class WeightNorm(Stateful):
    def __init__(self, param: Parameter, dim=0, epsilon=1e-8):
        self.dim = dim
        self.epsilon = epsilon

        self.v = Parameter(param.master.copy(), requires_grad=True)
        g = ops.norm(self.v, axis=dim, keepdims=True)
        self.g = Parameter(g, requires_grad=True)

        param.normalization = self

    def state_dict(self):
        return {"v_master_data": self.v.master.data,
                "v_master_grad": self.v.master.grad,
                "g_master_data": self.g.master.data,
                "g_master_grad": self.g.master.grad}
    
    def load_state_dict(self, state):
        if "v_master_data" in state:
            self.v.master.data[...] = state["v_master_data"]
        v_grad = state.get("v_master_grad", None)
        if v_grad is not None:
            self.v.master.grad = v_grad
        if "g_master_data" in state:
            self.g.master.data[...] = state["g_master_data"]
        g_grad = state.get("g_master_grad", None)
        if g_grad is not None:
            self.g.master.grad = g_grad

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