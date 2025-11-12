import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, Dense
from LunarLearn.core import Parameter, ops
from LunarLearn.nn import Stateful

xp = backend.xp

class LoRAParameter(Parameter):
    def __init__(self,
               param: Parameter,
               rank: int = 8,
               alpha: float = 16.0,
               keep_prob: float = 1.0
        ):
        super().__init__(xp.zeros((1,1)), requires_grad=False)
        self.param = param
        self.scaling = alpha / rank
        self.keep_prob = keep_prob

        d_out, d_in = param.shape

        A = xp.random.randn(d_in, rank) * 0.01
        B = xp.zeros((rank, d_out))

        self.A = Parameter(A, requires_grad=True)
        self.B = Parameter(B, requires_grad=True)

        self._state_fields = [
            "scaling",
            "keep_prob"
        ]

    def _named_state_items(self):
        for name, v in self.__dict__.items():
            if isinstance(v, Stateful):
                yield name, v

    def state_dict(self):
        out = {"_type": self.__class__.__name__}
        for name, obj in self._named_state_items():
            out[name] = obj.state_dict()
        for name in self._state_fields:
            val = getattr(self, name, None)
            if val is not None:
                out[name] = val
        return out

    def load_state_dict(self, state):
        for name, obj in self._named_state_items():
            if name in state:
                obj.load_state_dict(state[name])

        for name in self._state_fields:
            if name in state:
                setattr(self, name, state[name])

    def parameters(self, with_layer: bool = False):
        return [p for _, p in self.named_parameters(with_layer=with_layer)]

    def named_parameters(self, prefix: str = ""):
        params = []
        for name, v in self.__dict__.items():
            child_prefix = f"{prefix}.{name}"
            if isinstance(v, Parameter):
                for n, p in v.named_parameters(prefix=child_prefix):
                    params.append((n, p))
        return params

    def to_compute(self):
        A = self.A.to_compute()
        B = self.B.to_compute()
        param = self.param.to_compute()
        delta = ops.matmul(A, B) * self.scaling
        return param + delta

    def forward(self, x):
        A = self.A.to_compute()
        B = self.B.to_compute()
        param = self.param.to_compute()
        out = ops.matmul(x, param)
        if self.keep_prob < 1:
            x = ops.dropout(x, self.keep_prob)
        delta = ops.matmul(ops.matmul(x, A), B) * self.scaling
        return out + delta
    

def apply_lora(
        model,
        rank: int = 8,
        alpha: float = 16.0,
        target_attributes: list = ["Wq", "Wk", "Wv", "Wo"],
        keep_prob: float = 1.0
):
    
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        for attr in target_attributes:
            if hasattr(module, attr):
                param = getattr(module, attr)
                if isinstance(param, Parameter):
                    lora_param = LoRAParameter(param, rank=rank, alpha=alpha, keep_prob=keep_prob)
                    setattr(module, attr, lora_param)

    return model 