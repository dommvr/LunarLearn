from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, BatchNorm2D, Conv2D
from LunarLearn.core import Tensor, ops


class ASPPConv(BaseLayer):
    def __init__(self, filters, kernel_size, dilation, norm_layer=BatchNorm2D, activation="relu"):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)
        padding = 0 if kernel_size == 1 else dilation

        self.conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm = norm_layer()
        self.activation = get_activation(activation)

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return(x)
    

class ASPP(BaseLayer):
    """
    Atrous Spatial Pyramid Pooling:
    - 1x1 conv branch
    - k 3x3 atrous conv branches (different dilation rates)
    - concat -> 1x1 projection
    """
    def __init__(self, filters, atrous_rates=(6, 12, 18), norm_layer=BatchNorm2D, activation="relu"):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)
        self.branches = ModuleList()

        self.branches.append(
            ASPPConv(filters, kernel_size=1, dilation=1,
                     norm_layer=norm_layer, activation=activation)
        )

        for r in atrous_rates:
            self.branches.append(
                ASPPConv(filters, kernel_size=3, dilation=r,
                         norm_layer=norm_layer, activation=activation)
            )

        self.project_conv = Conv2D(filters, kernel_size=1, strides=1)
        self.project_norm = BatchNorm2D()
        self.project_act = get_activation(activation)

    def forward(self, x: Tensor):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))

        # concat along channels
        x = ops.concatenate(outs, axis=1)

        x = self.project_conv(x)
        x = self.project_norm(x)
        x = self.project_act(x)
        return x


class DeepLabV3(Module):
    def __init__(self,
                 num_classes,
                 filters=64,
                 depth=4,
                 atrous_rates=(6, 12, 18),
                 norm_layer=BatchNorm2D,
                 activation="relu",
                 final_activation=None,
                 pretrained=False):
        from LunarLearn.train.models.segmentation.unet import DownBlock, DoubleConv
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        self.encoder = ModuleList([
            DownBlock(filters * (2 ** i), norm_layer=norm_layer, activation=activation) for i in range(depth)
        ])

        bottleneck_filters = filters * (2 ** depth)
        self.bottleneck = DoubleConv(bottleneck_filters, norm_layer=norm_layer, activation=activation)

        self.aspp = ASPP(
            filters=bottleneck_filters,
            atrous_rates=atrous_rates,
            norm_layer=norm_layer,
            activation=activation
        )

        self.final_conv = Conv2D(filters=num_classes, kernel_size=1, strides=1)
        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor):
        input_h, input_w = x.shape[2], x.shape[3]
        for block in self.encoder:
            x, _ = block(x)

        x = self.bottleneck(x)
        x = self.aspp(x)
        x = self.final_conv(x)

        feat_h, feat_w = x.shape[2], x.shape[3]
        scale_h = input_h // feat_h
        scale_w = input_w // feat_w
        scale_factor = scale_h

        x = ops.upsample(x, scale_factor)
        out = self.final_act(x)
        return out
    

class DeepLabV3Plus(Module):
    def __init__(self,
                 num_classes,
                 filters=64,
                 depth=4,
                 atrous_rates=(6, 12, 18),
                 norm_layer=BatchNorm2D,
                 activation="relu",
                 final_activation=None,
                 low_level_filters=48,
                 pretrained=False):
        from LunarLearn.train.models.segmentation.unet import DownBlock, DoubleConv
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        self.num_classes = num_classes
        self.depth = depth

        self.encoder = ModuleList([
            DownBlock(filters * (2 ** i), norm_layer=norm_layer, activation=activation)
            for i in range(depth)
        ])

        bottleneck_filters = filters * (2 ** depth)
        self.bottleneck = DoubleConv(bottleneck_filters,
                                     norm_layer=norm_layer,
                                     activation=activation)
        
        self.aspp = ASPP(filters=bottleneck_filters,
                         atrous_rates=atrous_rates,
                         norm_layer=norm_layer,
                         activation=activation)
        
        self.low_level_conv = Conv2D(filters=low_level_filters,
                                     kernel_size=1,
                                     strides=1)
        self.low_level_norm = norm_layer()
        self.low_level_act = get_activation(activation)

        self.decoder_conv1 = Conv2D(filters=filters,
                                    kernel_size=3,
                                    strides=1,
                                    padding=1)
        self.decoder_norm1 = norm_layer()
        self.decoder_act1 = get_activation(activation)

        self.decoder_conv2 = Conv2D(filters=filters,
                                    kernel_size=3,
                                    strides=1,
                                    padding=1)
        self.decoder_norm2 = norm_layer()
        self.decoder_act2 = get_activation(activation)

        # Final classifier (on decoder output)
        self.final_conv = Conv2D(filters=num_classes,
                                 kernel_size=1,
                                 strides=1)
        
        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor):
        input_h, input_w = x.shape[2], x.shape[3]

        # ----- Encoder -----
        low_level = None
        for idx, block in enumerate(self.encoder):
            x, skip = block(x)  # DownBlock returns (downsampled, skip)
            if idx == 0:
                # use first high-res skip as low-level feature
                low_level = skip

        # ----- Bottleneck + ASPP -----
        x = self.bottleneck(x)
        x = self.aspp(x)  # shape: (B, bottleneck_filters, H_aspp, W_aspp)

        # ----- Upsample ASPP to low-level spatial size -----
        low_h, low_w = low_level.shape[2], low_level.shape[3]
        feat_h, feat_w = x.shape[2], x.shape[3]
        scale_h = low_h // feat_h
        scale_w = low_w // feat_w
        # assuming OS is consistent so scale_h == scale_w
        x = ops.upsample(x, scale_factor=scale_h)

        # ----- Project low-level feature -----
        low = self.low_level_conv(low_level)
        low = self.low_level_norm(low)
        low = self.low_level_act(low)

        # ----- Concatenate & decoder -----
        x = ops.concatenate([x, low], axis=1)

        x = self.decoder_conv1(x)
        x = self.decoder_norm1(x)
        x = self.decoder_act1(x)

        x = self.decoder_conv2(x)
        x = self.decoder_norm2(x)
        x = self.decoder_act2(x)

        # ----- Final classifier & upsample to input size -----
        x = self.final_conv(x)

        feat_h, feat_w = x.shape[2], x.shape[3]
        scale_h = input_h // feat_h
        scale_w = input_w // feat_w
        x = ops.upsample(x, scale_factor=scale_h)

        out = self.final_act(x)
        return out