from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor

class ModuleList(BaseLayer):
    def __init__(self, modules=None):
        super().__init__(trainable=True)
        # store list of submodules (EncoderBlock, DecoderBlock, etc.)
        self.modules = list(modules) if modules is not None else []

    def append(self, module):
        """Add a new submodule to the list."""
        self.modules.append(module)

    def extend(self, modules):
        """Add multiple submodules."""
        self.modules.extend(modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Sequentially pass x through all modules."""
        for m in self.modules:
            x = m(x, *args, **kwargs)
        return x

    def parameters(self, with_layer=False):
        """Collect parameters recursively from all submodules."""
        params = []
        for m in self.modules:
            params.extend(m.parameters(with_layer=with_layer))
        return params

    def named_parameters(self, prefix: str = "", with_layer: bool = False):
        """Same as parameters(), but with names included."""
        params = []
        for i, m in enumerate(self.modules):
            sub_prefix = f"{prefix}block{i}."
            params.extend(m.named_parameters(prefix=sub_prefix, with_layer=with_layer))
        return params
