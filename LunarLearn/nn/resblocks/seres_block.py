import LunarLearn.backend as backend
from LunarLearn.ResBlocks.BaseResBlock import BaseResBlock
from LunarLearn.tensor import Tensor
from LunarLearn.layers import Conv2D, BatchNorm2D, Dense, GlobalAveragePool2D

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class SEResBlock(BaseResBlock):
    def __init__(self, filters, reduction=16, strides=1, norm_layer=BatchNorm2D, activation="relu"):
        super().__init__(filters, strides=strides, activation=activation)
        self.conv1 = Conv2D(filters, kernel_size=3, strides=strides, padding="same")
        self.norm1 = norm_layer() if norm_layer else None
        self.conv2 = Conv2D(filters, kernel_size=3, strides=1, padding="same")
        self.norm2 = norm_layer() if norm_layer else None

        # --- SE attention layers ---
        self.global_pool = GlobalAveragePool2D()
        self.fc1 = Dense(filters // reduction, activation="relu")
        self.fc2 = Dense(filters, activation="sigmoid")

        self.shortcut = None

    def forward(self, x: Tensor) -> Tensor:
        from LunarLearn.activations import get_activation

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

        # Squeeze & Excitation
        se = self.global_pool(out)                # (B, C)
        se = self.fc1(se)                         # (B, C//r)
        se = self.fc2(se)                         # (B, C)
        se = se.reshape(out.shape[0], out.shape[1], 1, 1)  # (B, C, 1, 1)
        out = out * se                            # channel reweighting

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