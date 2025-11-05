import LunarLearn.backend as backend
from LunarLearn.ResBlocks.BaseResBlock import BaseResBlock
from LunarLearn.tensor import Tensor
from LunarLearn.layers import Conv2D, BatchNorm2D

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class PreActResBlock(BaseResBlock):
    def __init__(self, filters, strides=1, norm_layer=BatchNorm2D, activation="relu"):
        super().__init__(filters=filters, strides=strides, activation=activation)

        self.norm1 = norm_layer() if norm_layer else None
        self.conv1 = Conv2D(filters, kernel_size=3, strides=strides, padding="same")
        self.norm2 = norm_layer() if norm_layer else None
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding="same")

    def forward(self, x: Tensor) -> Tensor:
        from LunarLearn.activations import get_activation
        activation = get_activation(self.activation)

        # Create shortcut path if not built yet
        if self.shortcut is None:
            self._make_shortcut(x.shape)

        # Identity branch
        identity = self.shortcut(x)

        out = x
        if self.norm1:
            out = self.norm1(out)
        out = activation(out)
        out = self.conv1(out)
        if self.norm2:
            out = self.norm2(out)
        out = activation(out)
        out = self.conv2(out)

        # Residual fusion
        if callable(self.use_shortcut):
            out = self.use_shortcut(out, identity)
        elif self.use_shortcut:
            out += identity

        # Mixed precision casting
        if MIXED_PRECISION:
            out = out.astype(C_DTYPE, copy=False)
        else:
            out = out.astype(DTYPE, copy=False)

        return out