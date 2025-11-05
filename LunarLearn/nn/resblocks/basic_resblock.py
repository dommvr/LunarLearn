from LunarLearn.ResBlocks.BaseResBlock import BaseResBlock
from LunarLearn.tensor import Tensor
from LunarLearn.layers import Conv2D, BatchNorm2D

class BasicResBlock(BaseResBlock):
    def __init__(self, filters, strides=1, norm_layer=BatchNorm2D, activation="relu"):
        super().__init__(filters=filters, strides=strides, activation=activation)
        self.conv1 = Conv2D(filters, kernel_size=3, strides=strides, padding="same")
        self.norm1 = norm_layer() if norm_layer else None
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.norm2 = norm_layer() if norm_layer else None

    def _forward_main(self, x: Tensor) -> Tensor:
        from LunarLearn.activations import get_activation
        activation = get_activation(self.activation)

        out = self.conv1(x)
        if self.norm1:
            out = self.norm1(out)
        out = activation(out)
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)

        return out