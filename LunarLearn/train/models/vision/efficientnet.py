import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, Conv2D, GlobalAveragePool2D, Dense, BatchNorm2D, Dropout, DropPath
from LunarLearn.core import Tensor

xp = backend.xp


class SEBlock(BaseLayer):
    def __init__(self, in_channels, se_ratio=0.25, activation="swish"):
        super().__init__(trainable=True)
        reduce = max(1, int(in_channels * se_ratio))

        self.pool = GlobalAveragePool2D()
        self.fc1 = Dense(nodes=reduce, activation=activation)
        self.fc2 = Dense(nodes=in_channels, activation="sigmoid")

    def forward(self, x: Tensor):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = s.reshape(s.shape[0], s.shape[1], 1, 1)
        out = x * s
        return out
    

class MBConv(BaseLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 strides=1,
                 expand_ratio=6,
                 se_ratio=0.25,
                 norm_layer=BatchNorm2D,
                 activation="swish",
                 keep_prob_dp=1.0):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)

        self.use_residual = (strides == 1 and in_channels == out_channels)

        self.activation = get_activation(activation)

        mid_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = Conv2D(filters=mid_channels,
                                      kernel_size=1,
                                      strides=1,
                                      padding=0,
                                      bias=False)
            self.expand_norm = norm_layer()
        else:
            self.expand_conv = None
            self.expand_norm = None

        self.dw_conv = Conv2D(filters=mid_channels,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding="same",
                              groups=mid_channels,
                              bias=False)
        self.dw_norm = norm_layer()

        self.se = SEBlock(in_channels=mid_channels,
                          se_ratio=se_ratio,
                          activation=activation)
        
        self.project_conv = Conv2D(filters=out_channels,
                                   kernel_size=1,
                                   strides=1,
                                   padding=0,
                                   bias=False)
        self.project_norm = norm_layer()

        self.drop_path = DropPath(keep_prob_dp)

    def forward(self, x: Tensor):
        identity = x

        if self.expand_conv is not None:
            x = self.expand_conv(x)
            x = self.expand_norm(x)
            x = self.activation(x)

        x = self.dw_conv(x)
        x = self.dw_norm(x)
        x = self.activation(x)

        x = self.se(x)

        x = self.project_conv(x)
        x = self.project_norm(x)

        if self.use_residual:
            x = self.drop_path(x)
            x = x + identity
        return x
    

class EfficientNet(Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 depth_mult=1.0,
                 keep_prob=0.8,
                 keep_prob_dp=0.8,
                 norm_layer=BatchNorm2D,
                 activation="swish",
                 final_activation=None,
                 pretrained=False):
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        def round_filters(filters, width_mult=width_mult, depth_divisor=8, min_depth=None):
            if not width_mult:
                return filters
            filters *= width_mult
            min_depth = min_depth or depth_divisor
            new_filters = max(min_depth,
                            int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
            # avoid going down by >10%
            if new_filters < 0.9 * filters:
                new_filters += depth_divisor
            return int(new_filters)

        def round_repeats(repeats):
            return int(xp.ceil(repeats * depth_mult))
        
        # ----- MBConv configuration (EfficientNet-B0) -----
        # (expansion, kernel, out_c, num_blocks, stride)
        blocks_cfg = [
            (1, 3, 16, 1, 1),
            (6, 3, 24, 2, 2),
            (6, 5, 40, 2, 2),
            (6, 3, 80, 3, 2),
            (6, 5, 112, 3, 1),
            (6, 5, 192, 4, 2),
            (6, 3, 320, 1, 1),
        ]
        
        total_blocks = sum(round_repeats(n) for (_, _, _, n, _) in blocks_cfg)
        block_idx = 0
        
        # ----- Stem -----
        out_channels = round_filters(32)
        self.stem_conv = Conv2D(filters=out_channels,
                                kernel_size=3,
                                strides=2,
                                padding=1)
        self.stem_norm = norm_layer()
        self.activation = get_activation(activation)

        in_channels = out_channels

        self.blocks = ModuleList()
        for (exp, k, c, n, s) in blocks_cfg:
            out_c = round_filters(c)
            repeats = round_repeats(n)

            for i in range(repeats):
                stride = s if i == 0 else 1

                # linear schedule from keep_prob=1.0 -> keep_prob_dp
                if keep_prob_dp < 1.0:
                    # drop_connect_rate for last block = 1 - keep_prob_dp
                    drop_rate_max = 1.0 - keep_prob_dp
                    drop_rate = drop_rate_max * block_idx / max(1, total_blocks - 1)
                    keep_prob_block = 1.0 - drop_rate
                else:
                    keep_prob_block = 1.0

                self.blocks.append(
                    MBConv(in_channels=in_channels,
                           out_channels=out_c,
                           kernel_size=k,
                           strides=stride,
                           expand_ratio=exp,
                           se_ratio=0.25,
                           norm_layer=norm_layer,
                           activation=activation,
                           keep_prob_dp=keep_prob_block)
                )

                in_channels = out_c
                block_idx += 1

        # ----- Head -----
        head_channels = round_filters(1280)
        self.head_conv = Conv2D(filters=head_channels,
                                kernel_size=1,
                                strides=1,
                                padding=0)
        self.head_norm = norm_layer()

        self.global_pool = GlobalAveragePool2D()
        self.dropout = Dropout(keep_prob)
        self.fc = Dense(num_classes)

        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor):
        x = self.stem_conv(x)
        x = self.stem_norm(x)
        x = self.activation(x)

        x = self.blocks(x)

        x = self.head_conv(x)
        x = self.head_norm(x)
        x = self.activation(x)

        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        out = self.final_act(x)
        return out
    

class EfficientNetB0(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.0,
                         depth_mult=1.0,
                         pretrained=pretrained,
                         **kwargs)
        

class EfficientNetB1(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.0,
                         depth_mult=1.1,
                         pretrained=pretrained,
                         **kwargs)


class EfficientNetB2(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.1,
                         depth_mult=1.2,
                         pretrained=pretrained,
                         **kwargs)


class EfficientNetB3(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.2,
                         depth_mult=1.4,
                         pretrained=pretrained,
                         **kwargs)
        

class EfficientNetB4(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.4,
                         depth_mult=1.8,
                         pretrained=pretrained,
                         **kwargs)
        

class EfficientNetB5(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.6,
                         depth_mult=2.2,
                         pretrained=pretrained,
                         **kwargs)
        

class EfficientNetB6(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=1.8,
                         depth_mult=2.6,
                         pretrained=pretrained,
                         **kwargs)
        

class EfficientNetB7(EfficientNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(num_classes=num_classes,
                         width_mult=2.0,
                         depth_mult=3.1,
                         pretrained=pretrained,
                         **kwargs)