from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor


class Identity(BaseLayer):
    def __init__(self):
        super().__init__(trainable=False)
    
    def forward(self, x: Tensor) -> Tensor:
        return x