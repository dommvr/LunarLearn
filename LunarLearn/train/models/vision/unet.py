from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, Conv2D, BatchNorm2D, Conv2DTranspose, MaxPool2D, Activation
from LunarLearn.core import ops

class DoubleConv(BaseLayer):
    def __init__(self, filters, norm_layer=BatchNorm2D, activation="relu"):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)
        self.conv1 = Conv2D(filters, strides=1)
        self.norm1 = norm_layer()
        self.conv2 = Conv2D(filters, strides=1)
        self.norm2 = norm_layer()

        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        out = self.activation(x)
        return out
    
class DownBlock(BaseLayer):
    def __init__(self, filters, norm_layer=BatchNorm2D, activation="relu"):
        super().__init__(trainable=True)
        self.doubleconv = DoubleConv(filters, norm_layer, activation)
        self.pool = MaxPool2D(pool_size=2)

    def forward(self, x):
        skip = self.doubleconv(x)
        out = self.pool(skip)
        return out, skip

class UpBlock(BaseLayer):
    def __init__(self, filters, norm_layer=BatchNorm2D, activation="relu", mode="bilinear"):
        super().__init__(trainable=True)
        self.bilinear = ModuleList(Conv2D(filters, kernel_size=3, strides=1, padding=1),
                                   norm_layer(),
                                   Activation(activation)
                                   ) if mode == "bilinear" else None
        self.convtrans = Conv2DTranspose(filters, kernel_size=2, strides=2) if mode == "transpose" else None
        self.doubleconv = DoubleConv(filters, norm_layer, activation)

    def forward(self, x, skip):
        if self.convtrans is not None:
            x = self.convtrans(x)
        else:
            x = ops.upsample(x, scale_factor=2, mode="bilinear")
            x = self.bilinear(x)
        x = self.concat([x, skip])
        x = ops.concatenate([x, skip], axis=1)
        out = self.doubleconv(x)
        return out

class UNet(Module):
    def __init__(self, num_classes, filters, depth, mode, norm_layer, activation, final_activation, pretrained=False):
        from LunarLearn.nn.activations import get_activation
        self.encoder = ModuleList([DownBlock(filters*(2**i), norm_layer, activation) for i in range(depth)])
        self.bootleneck = DoubleConv(filters*(2**depth), norm_layer, activation)
        self.decoder = ModuleList([UpBlock(filters*(2**(i+1)), norm_layer, activation, mode) for i in range(depth)])
        self.final_conv = Conv2D(num_classes, kernel_size=1)

        if final_activation is not None:
            self.final_activation = get_activation(final_activation)
        else:
            if num_classes == 1:
                self.final_activation = get_activation("sigmoid")
            else:
                self.final_activation = get_activation("softmax")

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x):
        skips = []
        for block in self.encoder:
            x, skip = block(x)
            skips.append(skip)
        
        x = self.bootleneck(x)

        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
        
        x = self.final_conv(x)
        out = self.final_activation(x)
        return out