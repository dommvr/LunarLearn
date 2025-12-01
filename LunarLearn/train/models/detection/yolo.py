from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, Conv2D, BatchNorm2D
from LunarLearn.core import Tensor, ops


class ConvBNAct(BaseLayer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 use_norm=True,
                 activation="leaky_relu",   # YOLO usually uses LeakyReLU
                 alpha=0.1,                 # if your Activation supports slope
                 bias=False):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           bias=bias)
        self.norm = BatchNorm2D() if use_norm else None
        # If your Activation layer takes name + extra kwargs:
        self.activation = get_activation(activation)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x, alpha=self.alpha)
        return x
    

class DarknetResidualBlock(BaseLayer):
    """
    Simple 2-layer residual: Conv(1x1) -> Conv(3x3) + skip
    """
    def __init__(self, channels):
        super().__init__(trainable=True)
        self.conv1 = ConvBNAct(filters=channels // 2,
                               kernel_size=1,
                               strides=1,
                               padding=0)
        self.conv2 = ConvBNAct(filters=channels,
                               kernel_size=3,
                               strides=1,
                               padding=1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + identity


class Darknet53Backbone(Module):
    """
    Darknet-53 backbone returning 3 feature maps:
    - C3: 256 channels (52x52 for 416x416 input)
    - C4: 512 channels (26x26)
    - C5: 1024 channels (13x13)
    """
    def __init__(self):
        super().__init__()

        self.stem = ConvBNAct(filters=32,
                              kernel_size=3,
                              strides=1,
                              padding=1)

        # (64, 1 block)
        self.down1 = ConvBNAct(filters=64,
                               kernel_size=3,
                               strides=2,
                               padding=1)
        self.stage1 = ModuleList([DarknetResidualBlock(64)])

        # (128, 2 blocks)
        self.down2 = ConvBNAct(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding=1)
        self.stage2 = ModuleList([DarknetResidualBlock(128) for _ in range(2)])

        # (256, 8 blocks) -> C3
        self.down3 = ConvBNAct(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding=1)
        self.stage3 = ModuleList([DarknetResidualBlock(256) for _ in range(8)])

        # (512, 8 blocks) -> C4
        self.down4 = ConvBNAct(filters=512,
                               kernel_size=3,
                               strides=2,
                               padding=1)
        self.stage4 = ModuleList([DarknetResidualBlock(512) for _ in range(8)])

        # (1024, 4 blocks) -> C5
        self.down5 = ConvBNAct(filters=1024,
                               kernel_size=3,
                               strides=2,
                               padding=1)
        self.stage5 = ModuleList([DarknetResidualBlock(1024) for _ in range(4)])

    def forward(self, x: Tensor):
        x = self.stem(x)

        x = self.down1(x)
        x = self.stage1(x)

        x = self.down2(x)
        x = self.stage2(x)

        x = self.down3(x)
        x = self.stage3(x)
        c3 = x  # 256-ch

        x = self.down4(x)
        x = self.stage4(x)
        c4 = x  # 512-ch

        x = self.down5(x)
        x = self.stage5(x)
        c5 = x  # 1024-ch

        return c3, c4, c5
    

class YoloConvBlock(BaseLayer):
    """
    The 5-layer conv stack used in YOLOv3 heads:
    C -> C' -> ... -> C'
    """
    def __init__(self, out_channels):
        super().__init__(trainable=True)
        # typical pattern: 1x1,3x3,1x1,3x3,1x1
        mid = out_channels

        self.layers = ModuleList([
            ConvBNAct(filters=mid,   kernel_size=1, strides=1, padding=0),
            ConvBNAct(filters=mid*2, kernel_size=3, strides=1, padding=1),
            ConvBNAct(filters=mid,   kernel_size=1, strides=1, padding=0),
            ConvBNAct(filters=mid*2, kernel_size=3, strides=1, padding=1),
            ConvBNAct(filters=mid,   kernel_size=1, strides=1, padding=0),
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x


class YoloHead(BaseLayer):
    """
    One YOLO prediction head operating on a feature map.
    Returns:
    - pred: (B, A*(5+num_classes), H, W)
    - route: feature map for next upsampling (if needed)
    """
    def __init__(self,
                 num_classes,
                 num_anchors,
                 out_channels_mid):
        super().__init__(trainable=True)
        self.yolo_conv = YoloConvBlock(out_channels_mid)

        # prediction conv: no BN, no activation
        self.pred_conv = Conv2D(
            filters=num_anchors * (num_classes + 5),
            kernel_size=1,
            strides=1,
            padding=0,
            bias=True
        )

    def forward(self, x: Tensor):
        x = self.yolo_conv(x)
        route = x
        pred = self.pred_conv(x)
        return pred, route
    

class YOLOv3(Module):
    """
    YOLOv3-style detector.
    Outputs list of 3 prediction tensors:
    [
      P_small:  (B, A*(5+num_classes), H/8,  W/8),
      P_medium: (B, A*(5+num_classes), H/16, W/16),
      P_large:  (B, A*(5+num_classes), H/32, W/32),
    ]
    """
    def __init__(self,
                 num_classes,
                 num_anchors_per_scale=3,
                 small_first=True):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors_per_scale = num_anchors_per_scale
        self.small_first = small_first

        # Backbone
        self.backbone = Darknet53Backbone()

        # Large scale head (C5: 1024 channels)
        self.head_large = YoloHead(
            num_classes=num_classes,
            num_anchors=num_anchors_per_scale,
            out_channels_mid=512
        )

        # Up-conv from large to medium
        self.conv_upsample1 = ConvBNAct(filters=256, kernel_size=1, strides=1, padding=0)

        # Medium scale head (C4: 512 + upsampled 256 = 768)
        self.head_medium = YoloHead(
            num_classes=num_classes,
            num_anchors=num_anchors_per_scale,
            out_channels_mid=256
        )

        # Up-conv from medium to small
        self.conv_upsample2 = ConvBNAct(filters=128, kernel_size=1, strides=1, padding=0)

        # Small scale head (C3: 256 + upsampled 128 = 384)
        self.head_small = YoloHead(
            num_classes=num_classes,
            num_anchors=num_anchors_per_scale,
            out_channels_mid=128
        )

    def forward(self, x: Tensor):
        # Backbone features
        c3, c4, c5 = self.backbone(x)   # (256, 512, 1024)

        # Large scale (13x13 for 416 input)
        p_large, route_large = self.head_large(c5)

        # Medium scale
        x_mid = self.conv_upsample1(route_large)
        x_mid = ops.upsample(x_mid, scale_factor=2, mode="nearest")
        x_mid = ops.concatenate([x_mid, c4], axis=1)
        p_medium, route_medium = self.head_medium(x_mid)

        # Small scale
        x_small = self.conv_upsample2(route_medium)
        x_small = ops.upsample(x_small, scale_factor=2, mode="nearest")
        x_small = ops.concatenate([x_small, c3], axis=1)
        p_small, _ = self.head_small(x_small)

        if self.small_first:
            return [p_small, p_medium, p_large]
        else:
            return [p_large, p_medium, p_small]