import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, Embedding
from LunarLearn.core import Parameter, ops

xp = backend.xp

class PromptEmbedding(BaseLayer):
    def __init__(self, layer, n_tokens, d_model):
        super().__init__(trainable=True)
        self.layer = layer
        emb = xp.random.randn(n_tokens, d_model)
        self.emb = Parameter(emb, requires_grad=True)

    def forward(self, x):
        emb = self.emb.to_compute()
        pos_emb = self.layer(x)
        out = ops.concatenate([emb, pos_emb], axis=1)
        return out
    

def add_prompt(model, n_tokens, d_model):
    for p in model.named_parameters():
        p.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, Embedding):
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name)
            prompt_module = PromptEmbedding(module, n_tokens=n_tokens, d_model=d_model)
            setattr(parent, module_name, prompt_module)
            break
    
    return model