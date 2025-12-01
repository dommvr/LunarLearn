from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, Conv2D, BatchNorm2D, GlobalAveragePool2D, Dense, Dropout
from LunarLearn.core import Tensor

class InvertedResidual(BaseLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=1,
                 expand_ratio=6,
                 norm_layer=BatchNorm2D,
                 activation="relu6"):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)

        assert strides in [1, 2], "MobileNetV2 only uses stride 1 or 2 in blocks."
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = (strides == 1 and in_channels == out_channels)

        self.activation = get_activation(activation)

        # 1x1 pointwise conv (expansion)
        if expand_ratio != 1:
            self.pw_conv1 = Conv2D(filters=hidden_dim,
                                   kernel_size=1,
                                   strides=1,
                                   padding=0,
                                   bias=False)
            self.pw_norm1 = norm_layer()
        else:
            self.pw_conv1 = None
            self.pw_norm1 = None
            hidden_dim = in_channels  # to be explicit

        # 3x3 depthwise conv
        self.dw_conv = Conv2D(filters=hidden_dim,
                              kernel_size=3,
                              strides=strides,
                              padding=1,
                              groups=hidden_dim,
                              bias=False)
        self.dw_norm = norm_layer()

        # 1x1 pointwise linear (projection)
        self.pw_conv2 = Conv2D(filters=out_channels,
                               kernel_size=1,
                               strides=1,
                               padding=0,
                               bias=False)
        self.pw_norm2 = norm_layer()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # expansion
        if self.pw_conv1 is not None:
            x = self.pw_conv1(x)
            x = self.pw_norm1(x)
            x = self.activation(x)

        # depthwise
        x = self.dw_conv(x)
        x = self.dw_norm(x)
        x = self.activation(x)

        # projection
        x = self.pw_conv2(x)
        x = self.pw_norm2(x)

        # residual
        if self.use_residual:
            x = x + identity

        return x


class MobileNetV2(Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 dropout_keep=0.2,
                 norm_layer=BatchNorm2D,
                 activation="relu6",
                 final_activation=None,
                 pretrained=False):
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        def _make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return int(new_v)

        inverted_residual_cfg = [
            # t,  c,   n, s
            [1,  16,  1, 1],
            [6,  24,  2, 2],
            [6,  32,  3, 2],
            [6,  64,  4, 2],
            [6,  96,  3, 1],
            [6, 160,  3, 2],
            [6, 320,  1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, 8)
        last_channel = _make_divisible(1280 * max(1.0, width_mult), 8)

        self.activation = get_activation(activation)

        # Stem: conv + BN + act
        self.stem_conv = Conv2D(filters=input_channel,
                                kernel_size=3,
                                strides=2,
                                padding=1,
                                bias=False)
        self.stem_norm = norm_layer()

        # Inverted residual blocks
        self.blocks = ModuleList()
        in_c = input_channel
        for t, c, n, s in inverted_residual_cfg:
            out_c = _make_divisible(c * width_mult, 8)
            for i in range(n):
                strides = s if i == 0 else 1
                self.blocks.append(
                    InvertedResidual(
                        in_channels=in_c,
                        out_channels=out_c,
                        strides=strides,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                        activation=activation,
                    )
                )
                in_c = out_c

        # Head
        self.head_conv = Conv2D(filters=last_channel,
                                kernel_size=1,
                                strides=1,
                                padding=0,
                                bias=False)
        self.head_norm = norm_layer()
        self.head_act = get_activation(activation)

        self.global_pool = GlobalAveragePool2D()
        self.dropout = Dropout(keep_prob=dropout_keep)
        self.fc = Dense(num_classes)

        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        x = self.activation(x)

        # Inverted residual stack
        x = self.blocks(x)

        # Head
        x = self.head_conv(x)
        x = self.head_norm(x)
        x = self.head_act(x)

        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        out = self.final_act(x)
        return out


class MobileNetV2_0_5(MobileNetV2):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=0.5,
                         pretrained=pretrained,
                         **kwargs)


class MobileNetV2_1_0(MobileNetV2):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.0,
                         pretrained=pretrained,
                         **kwargs)