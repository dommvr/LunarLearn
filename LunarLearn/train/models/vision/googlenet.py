from LunarLearn.nn import Module
from LunarLearn.nn.layers import BaseLayer, Conv2D, MaxPool2D, GlobalAveragePool2D, Dropout, Dense, BatchNorm2D
from LunarLearn.nn.inception import InceptionV1Block
from LunarLearn.core import Tensor


class ConvBlock(BaseLayer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 norm_layer=None,
                 activation="relu",
                 bias=True):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)
        
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           bias=bias)
        self.norm = norm_layer() if norm_layer is not None else None
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        out = self.activation(x)
        return out


class GoogLeNet(Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer=BatchNorm2D,
                 keep_prob=0.6,
                 activation="relu",
                 final_activation=None,
                 pretrained=False):
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        self.conv1 = ConvBlock(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding=3,
                               norm_layer=norm_layer,
                               activation=activation)
        self.pool1 = MaxPool2D(pool_size=3, strides=2, padding=1)

        self.conv2a = ConvBlock(filters=64,
                                kernel_size=1,
                                strides=1,
                                padding=0,
                                norm_layer=norm_layer,
                                activation=activation)
        self.conv2b = ConvBlock(filters=192,
                                kernel_size=3,
                                strides=1,
                                padding=1,
                                norm_layer=norm_layer,
                                activation=activation)
        self.pool2 = MaxPool2D(pool_size=3, strides=2, padding=1)

        self.inception3a = InceptionV1Block(f_1x1=64,
                                            f_3x3_reduce=96,
                                            f_3x3=128,
                                            f_5x5_reduce=16,
                                            f_5x5=32,
                                            f_pool_proj=32,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.inception3b = InceptionV1Block(f_1x1=128,
                                            f_3x3_reduce=128,
                                            f_3x3=192,
                                            f_5x5_reduce=32,
                                            f_5x5=96,
                                            f_pool_proj=64,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.pool3 = MaxPool2D(pool_size=3, strides=2, padding=1)

        self.inception4a = InceptionV1Block(f_1x1=192,
                                            f_3x3_reduce=96,
                                            f_3x3=208,
                                            f_5x5_reduce=16,
                                            f_5x5=48,
                                            f_pool_proj=64,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.inception4b = InceptionV1Block(f_1x1=160,
                                            f_3x3_reduce=112,
                                            f_3x3=224,
                                            f_5x5_reduce=24,
                                            f_5x5=64,
                                            f_pool_proj=64,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.inception4c = InceptionV1Block(f_1x1=128,
                                            f_3x3_reduce=128,
                                            f_3x3=256,
                                            f_5x5_reduce=24,
                                            f_5x5=64,
                                            f_pool_proj=64,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.inception4d = InceptionV1Block(f_1x1=112,
                                            f_3x3_reduce=144,
                                            f_3x3=288,
                                            f_5x5_reduce=32,
                                            f_5x5=64,
                                            f_pool_proj=64,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.inception4e = InceptionV1Block(f_1x1=256,
                                            f_3x3_reduce=160,
                                            f_3x3=320,
                                            f_5x5_reduce=32,
                                            f_5x5=128,
                                            f_pool_proj=128,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.pool4 = MaxPool2D(pool_size=3, strides=2, padding=1)

        self.inception5a = InceptionV1Block(f_1x1=256,
                                            f_3x3_reduce=160,
                                            f_3x3=320,
                                            f_5x5_reduce=32,
                                            f_5x5=128,
                                            f_pool_proj=128,
                                            norm_layer=norm_layer,
                                            activation=activation)
        self.inception5b = InceptionV1Block(f_1x1=384,
                                            f_3x3_reduce=192,
                                            f_3x3=384,
                                            f_5x5_reduce=48,
                                            f_5x5=128,
                                            f_pool_proj=128,
                                            norm_layer=norm_layer,
                                            activation=activation)
        
        self.global_pool = GlobalAveragePool2D()
        self.dropout = Dropout(keep_prob)
        self.fc = Dense(num_classes)
        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.fc(x)
        out = self.final_act(x)
        return out