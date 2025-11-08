from LunarLearn.nn.resblocks import BaseResBlock
from LunarLearn.core import Tensor
from LunarLearn.nn.layers import Conv2D, BatchNorm2D

class BottleneckResBlock(BaseResBlock):
    def __init__(self, filters, strides=1, norm_layer=BatchNorm2D, activation="relu", expansion=4):
        self.expansion = expansion
        expanded = filters * expansion
        super().__init__(expanded, strides=strides, activation=activation)

        reduced = filters

        self.conv1 = Conv2D(reduced, kernel_size=1, strides=1)
        self.norm1 = norm_layer() if norm_layer else None
        self.conv2 = Conv2D(reduced, kernel_size=3, strides=strides, padding="same")
        self.norm2 = norm_layer() if norm_layer else None
        self.conv3 = Conv2D(expanded, kernel_size=1, strides=1)
        self.norm3 = norm_layer() if norm_layer else None

    def _forward_main(self, x: Tensor) -> Tensor:
        from LunarLearn.nn.activations import get_activation
        activation = get_activation(self.activation)

        out = self.conv1(x)
        if self.norm1:
            out = self.norm1(out)
        out = activation(out)
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)
        out = activation(out)
        out = self.conv3(out)
        if self.norm3:
            out = self.norm3(out)

        return out