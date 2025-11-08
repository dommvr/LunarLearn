from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import ops

class Concat(BaseLayer):
    def __init__(self, axis=0):
        super().__init__(trainable=False)
        self.axis = axis

    def forward(self, x):
        return ops.concatenate(x, axis=self.axis)