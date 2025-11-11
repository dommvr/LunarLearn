import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, Dense
from LunarLearn.core import Parameter, ops

xp = backend.xp

class LoRA(BaseLayer):
    def __init__(self,
               layer: Dense,
               rank: int = 8,
               alpha: float = 16.0,
               keep_prob: float = 1.0
        ):

        super().__init__(trainable=True)
        self.layer = layer
        self.scaling = alpha / rank
        self.keep_prob = keep_prob

        d_out, d_in = layer.W.shape

        A = xp.random.randn(d_out, d_in) * 0.01
        B = xp.zeros((rank, d_in))

        self.A = Parameter(A, requires_grad=True)
        self.B = Parameter(B, requires_grad=True)

    def forward(self,  x):
        out = self.layer(x)

        if self.keep_prob < 1:
            out = ops.dropout(out, self.keep_prob)

        delta = ops.matmul(ops.matmul(x, self.B.T), self.A.T) * self.scaling
        return out + delta
    

def apply_lora(
        model,
        rank: int = 8,
        alpha: float = 16.0,
        target_modules: list = ["q_proj", "v_proj"],
        keep_prob: float = 1.0
):
    
    for p in model.parameters():
        p.requires_grad = False

    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, Dense):
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            lora_module = LoRA(module, rank=rank, alpha=alpha, keep_prob=keep_prob)
            setattr(parent, module_name, lora_module)

    return model 