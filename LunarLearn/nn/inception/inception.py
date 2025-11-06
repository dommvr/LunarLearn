from LunarLearn.nn.layers import BaseLayer
from LunarLearn.nn.inception import InceptionBranch
from LunarLearn.core import Tensor, ops

class Inception(BaseLayer):
    def __init__(self, *branches, axis=1):
        super().__init__(trainable=True)
        self.branches = []
        for b in branches:
            # Allow user to pass list of layers instead of InceptionBranch
            if isinstance(b, (list, tuple)):
                self.branches.append(InceptionBranch(*b))
            else:
                self.branches.append(b)
        self.axis = axis
        self.norm_layer = None

    def forward(self, x: Tensor) -> Tensor:
        out = [branch(x) for branch in self.branches]
        out = ops.concatenate(out, axis=self.axis)

        return out
    
    def _make_conv_layers(self, layers):
        from LunarLearn.nn.layers import Activation
        seq = []
        for layer in layers:
            seq.append(layer)
            if self.norm_layer:
                seq.append(self.norm_layer())
            if self.activation:
                seq.append(Activation(self.activation))

        return seq