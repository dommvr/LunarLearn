import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, Conv2D, LayerNorm, DropPath, Dense, Dropout, GlobalAveragePool2D
from LunarLearn.core import Tensor, Parameter

xp = backend.xp
DTYPE = backend.DTYPE


class ConvNeXtBlock(BaseLayer):
    def __init__(self,
                 dim,
                 layer_scale_init_value=1e-06,
                 keep_prob=1.0,
                 activation="gelu"):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)

        self.activation = get_activation(activation)

        # Depthwise 7x7 conv
        self.dw_conv = Conv2D(filters=dim,
                              kernel_size=7,
                              strides=1,
                              padding="same",
                              groups=dim,
                              bias=False)
        self.dw_norm = LayerNorm(epsilon=1e-6, axis=(1,))

        # Pointwise conv "MLP": dim -> 4*dim -> dim
        self.pw_conv1 = Conv2D(filters=4 * dim,
                               kernel_size=1,
                               strides=1,
                               padding=0,
                               bias=True)
        self.pw_conv2 = Conv2D(filters=dim,
                               kernel_size=1,
                               strides=1,
                               padding=0,
                               bias=True)
        
        # LayerScale gamma
        if layer_scale_init_value is not None and layer_scale_init_value > 0.0:
            gamma = layer_scale_init_value * xp.ones((1, dim, 1, 1), dtype=DTYPE)
            self.gamma = Parameter(gamma, requires_grad=True)
        else:
            self.gamma = None

        self.drop_path = DropPath(keep_prob)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # Depthwise
        x = self.dw_conv(x)
        x = self.dw_norm(x)

        # MLP
        x = self.pw_conv1(x)
        x = self.activation(x)
        x = self.pw_conv2(x)

        # LayerScale
        if self.gamma is not None:
            gamma = self.gamma.to_compute()
            x = gamma * x

        # DropPath + residual
        x = self.drop_path(x)
        out = identity + x
        return out
    

class ConvNeXtDownsample(BaseLayer):
    def __init__(self, out_channels):
        super().__init__(trainable=True)

        self.norm = LayerNorm(epsilon=1e-6, axis=(1,))
        self.conv = Conv2D(filters=out_channels,
                           kernel_size=2,
                           strides=2,
                           padding=0,
                           bias=True)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.conv(x)
        return x
    
class ConvNeXt(Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=1000,
                 depths=(3, 3, 9, 3),
                 dims=(96, 192, 384, 768),
                 keep_prob_dp=1.0,
                 layer_scale_init_value=1e-6,
                 activation="gelu",
                 head_keep_prob=0.8,
                 final_activation=None,
                 pretrained=False):
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        assert len(depths) == 4 and len(dims) == 4, "ConvNeXt expects 4 stages."

        self.stem = ModuleList([
            Conv2D(filters=dims[0],
                   kernel_size=4,
                   strides=4,
                   padding=0,
                   bias=False),
            LayerNorm(epsilon=1e-6, axis=(1,))
        ])

        # ----- Stages (stacks of ConvNeXtBlock) -----
        self.stages = ModuleList()
        total_blocks = sum(depths)
        block_idx = 0

        for stage_idx in range(4):
            stage_blocks = ModuleList()
            dim = dims[stage_idx]
            depth = depths[stage_idx]

            for _ in range(depth):
                if keep_prob_dp < 1.0:
                    drop_rate_max = 1.0 - keep_prob_dp
                    drop_rate = drop_rate_max * block_idx / max(1, total_blocks - 1)
                    keep_prob_block = 1.0 - drop_rate
                else:
                    keep_prob_block = 1.0

                stage_blocks.append(
                    ConvNeXtBlock(
                        dim=dim,
                        layer_scale_init_value=layer_scale_init_value,
                        keep_prob=keep_prob_block,
                        activation=activation
                    )
                )
                block_idx += 1

            self.stages.append(stage_blocks)
            if stage_idx < 3:
                self.stages.append(ConvNeXtDownsample(dims[stage_idx + 1]))

        # ----- Head -----
        self.head_norm = LayerNorm(epsilon=1e-6, axis=(1,))
        self.global_pool = GlobalAveragePool2D()
        self.dropout = Dropout(keep_prob=head_keep_prob)
        self.fc = Dense(num_classes)

        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)

        x = self.head_norm(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        out = self.final_act(x)
        return out


class ConvNeXtTiny(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(
            num_classes=num_classes,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            keep_prob_dp=0.9,
            layer_scale_init_value=1e-6,
            pretrained=pretrained,
            **kwargs
        )


class ConvNeXtSmall(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(
            num_classes=num_classes,
            depths=(3, 3, 27, 3),
            dims=(96, 192, 384, 768),
            keep_prob_dp=0.9,
            layer_scale_init_value=1e-6,
            pretrained=pretrained,
            **kwargs
        )


class ConvNeXtBase(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(
            num_classes=num_classes,
            depths=(3, 3, 27, 3),
            dims=(128, 256, 512, 1024),
            keep_prob_dp=0.9,
            layer_scale_init_value=1e-6,
            pretrained=pretrained,
            **kwargs
        )


class ConvNeXtLarge(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(
            num_classes=num_classes,
            depths=(3, 3, 27, 3),
            dims=(192, 384, 768, 1536),
            keep_prob_dp=0.8,
            layer_scale_init_value=1e-6,
            pretrained=pretrained,
            **kwargs
        )


class ConvNeXtXLarge(ConvNeXt):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(
            num_classes=num_classes,
            depths=(3, 3, 27, 3),
            dims=(256, 512, 1024, 2048),
            keep_prob_dp=0.7,
            layer_scale_init_value=1e-6,
            pretrained=pretrained,
            **kwargs
        )