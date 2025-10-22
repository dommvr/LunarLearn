from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor

class InceptionBranch(BaseLayer):
    def __init__(self, *layers):
        super().__init__(trainable=True)

        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out