import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.resblocks import BaseResBlock
from LunarLearn.core import Tensor
from LunarLearn.nn.layers import Conv2D, BatchNorm2D

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class ResNeXtBlock(BaseResBlock):
    def __init__(self, filters, groups=32, bottleneck_ratio=4, strides=1, norm_layer=BatchNorm2D, activation="relu"):
        super().__init__(filters, strides=strides, activation=activation)
        mid_channels = filters // bottleneck_ratio

        self.conv1 = Conv2D(mid_channels, kernel_size=1, strides=1, padding="same")
        self.norm1 = norm_layer() if norm_layer else None
        self.conv2 = Conv2D(mid_channels, kernel_size=3, strides=strides, padding="same", groups=groups)
        self.norm2 = norm_layer() if norm_layer else None
        self.conv3 = Conv2D(filters, kernel_size=1, strides=1, padding="same")
        self.norm3 = norm_layer() if norm_layer else None

        self.shortcut = None
        self.strides = strides

    def forward(self, x: Tensor) -> Tensor:
        from LunarLearn.nn.activations import get_activation

        # Create shortcut path if not built yet
        if self.shortcut is None:
            self._make_shortcut(x.shape)

        # Identity branch
        identity = self.shortcut(x)

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

        # Residual fusion
        if callable(self.use_shortcut):
            out = self.use_shortcut(out, identity)
        elif self.use_shortcut:
            out += identity

        # Final activation
        out = activation(out)

        # Mixed precision casting
        if MIXED_PRECISION:
            out = out.astype(C_DTYPE, copy=False)
        else:
            out = out.astype(DTYPE, copy=False)

        return out