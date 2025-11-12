import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, Dense
from LunarLearn.core import Parameter, ops

xp = backend.xp

class LoRAParameter(Parameter):
    def __init__(self,
               param: Parameter,
               rank: int = 8,
               alpha: float = 16.0,
               keep_prob: float = 1.0
        ):

        super().__init__(trainable=True)
        self.param = param
        self.scaling = alpha / rank
        self.keep_prob = keep_prob

        d_out, d_in = param.shape

        A = xp.random.randn(d_in, rank) * 0.01
        B = xp.zeros((rank, d_out))

        self.A = Parameter(A, requires_grad=True)
        self.B = Parameter(B, requires_grad=True)

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