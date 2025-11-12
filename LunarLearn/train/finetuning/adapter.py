from LunarLearn.nn.layers import BaseLayer, Dense
from LunarLearn.nn.activations import get_activation
from LunarLearn.core import ops

class Adapter(BaseLayer):
    def __init__(self, layer, d_model, bottleneck=64, keep_prob=1.0):
        super().__init__(trainable=True)
        self.layer = layer
        self.down = Dense(bottleneck)
        self.up = Dense(d_model)
        self.act = get_activation("gelu")
        self.keep_prob = keep_prob

    def forward(self, x, **kwargs):
        out = self.layer(x, **kwargs)
        adapter_out = self.down(x)
        adapter_out = self.act(adapter_out)
        adapter_out = ops.dropout(adapter_out, self.keep_prob, self.training)
        adapter_out = self.up(adapter_out)
        return out + adapter_out
    

def add_adapter(
        model,
        d_model: int,
        bottleneck: int = 64,
        keep_prob: float = 1.0,
        target_modules: list = [
            "mhattention",
            "self_attn",
            "cross_attn",
            "feedforward"
        ]
):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            adapter = Adapter(module, d_model, bottleneck, keep_prob)
            setattr(parent, module_name, adapter)

    return model