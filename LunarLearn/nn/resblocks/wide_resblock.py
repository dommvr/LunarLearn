from LunarLearn.nn.resblocks import BaseResBlock
from LunarLearn.nn.layers import Conv2D, BatchNorm2D
from LunarLearn.core import Tensor

class WideResBlock(BaseResBlock):
    """
    Wide Residual Block used in WideResNet architectures.

    Key differences from BasicResBlock:
        - Uses a widening factor to increase channel capacity.
        - Typically fewer layers but much wider, improving feature diversity.
        - Works well for medium-depth networks (e.g., WideResNet-28-10 on CIFAR).

    Args:
        filters (int): Base number of filters for the block.
        widen_factor (int): Factor to widen the number of filters (default: 2).
        strides (int): Convolution stride for the first conv layer.
        norm_layer (callable or None): Normalization layer (e.g., BatchNorm2D) or None to disable.
        activation (str): Activation function name (default: "relu").
    """
    def __init__(self, filters, widen_factor=2, strides=1, norm_layer=BatchNorm2D, activation="relu"):
        widened_filters = filters * widen_factor
        super().__init__(filters=widened_filters, strides=strides, activation=activation)

        self.conv1 = Conv2D(widened_filters, kernel_size=3, strides=strides, padding="same")
        self.norm1 = norm_layer() if norm_layer else None
        self.conv2 = Conv2D(widened_filters, kernel_size=3, strides=1, padding="same")
        self.norm2 = norm_layer() if norm_layer else None

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

        return out
