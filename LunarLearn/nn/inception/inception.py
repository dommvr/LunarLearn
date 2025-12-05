from LunarLearn.nn import ModuleList
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, ops


class Inception(BaseLayer):
    def __init__(self, *branches, axis=1):
        super().__init__(trainable=True)
        self.branches = []
        for b in branches:
            if isinstance(b, (list, tuple)):
                self.branches.append(ModuleList(b))
            else:
                self.branches.append(b)
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        out = [branch(x) for branch in self.branches]
        out = ops.concatenate(out, axis=self.axis)

        return out
    
    def _make_conv_layers(self, layers, norm_layer=None, activation="relu"):
        from LunarLearn.nn.layers import Activation
        seq = []
        for layer in layers:
            seq.append(layer)
            if norm_layer is not None:
                seq.append(norm_layer())
            if activation:
                seq.append(Activation(activation))

        return seq