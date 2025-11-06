import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer, Conv2D, Conv2DTranspose
from LunarLearn.core import Tensor

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class DepthwiseSeparableConv2DTranspose(BaseLayer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding=0,
        activation="linear",
        w_init="auto",
        uniform=False,
        gain=1,
        norm_layer=None
    ):
        super().__init__(trainable=True)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain
        self.norm_layer = norm_layer

        # These will be initialized on first forward call
        self.depthwise = None
        self.pointwise = None
        self.norm1 = None
        self.norm2 = None

    def initialize(self, input_shape):
        in_channels = input_shape[0]

        # Depthwise: groups = in_channels
        self.depthwise = Conv2DTranspose(
            filters=in_channels,  # same number of channels in depthwise
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation="linear",
            w_init=self.w_init,
            uniform=self.uniform,
            gain=self.gain,
            groups=in_channels
        )

        # Optional norm after depthwise
        self.norm1 = self.norm_layer() if self.norm_layer else None

        # Pointwise: 1x1 conv to mix channels
        self.pointwise = Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="linear",
            w_init=self.w_init,
            uniform=self.uniform,
            gain=self.gain
        )

        # Optional norm after pointwise
        self.norm2 = self.norm_layer() if self.norm_layer else None

        self.output_shape = (self.filters, None, None)  # Computed at runtime

    def forward(self, X: Tensor) -> Tensor:
        from LunarLearn.nn.activations import get_activation

        # Lazy init on first pass
        if self.depthwise is None or self.pointwise is None:
            self.initialize(X.shape[1:])

        activation = get_activation(self.activation)

        # Depthwise step
        out = self.depthwise(X)
        if self.norm1:
            out = self.norm1(out)
        out = activation(out)

        # Pointwise step
        out = self.pointwise(out)
        if self.norm2:
            out = self.norm2(out)
        out = activation(out)

        return out
