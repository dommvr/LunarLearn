import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class BaseResBlock(BaseLayer):
    def __init__(self, filters, strides=1, norm_layer=None, activation="relu", use_shortcut=True):
        super().__init__(trainable=True)
        self.filters = filters
        self.strides = strides
        self.norm_layer = norm_layer
        self.activation = activation

        self.main_path = []
        self.shortcut = None
        self.use_shortcut = use_shortcut

    def _make_shortcut(self, input_shape):
        from LunarLearn.layers import Conv2D

        in_channels = input_shape[1]
        out_channels = self.filters

        if in_channels != out_channels or self.strides != 1:
            self.shortcut = Conv2D(filters=out_channels,
                                   kernel_size=1,
                                   strides=self.strides
            )
        else:
            self.shortcut = lambda x: x


    def forward(self, x: Tensor) -> Tensor:
        from LunarLearn.activations import get_activation

        # Create shortcut path if not built yet
        if self.shortcut is None:
            self._make_shortcut(x.shape)

        # Identity branch
        identity = self.shortcut(x)

        # Main path
        out = self._forward_main(x)

        # Residual fusion
        if callable(self.use_shortcut):
            out = self.use_shortcut(out, identity)
        elif self.use_shortcut:
            out += identity

        # Final activation
        activation = get_activation(self.activation)
        out = activation(out)

        # Mixed precision casting
        if MIXED_PRECISION:
            out = out.astype(C_DTYPE, copy=False)
        else:
            out = out.astype(DTYPE, copy=False)

        return out
        
    def _forward_main(self, x: Tensor) -> Tensor:
        raise NotImplementedError