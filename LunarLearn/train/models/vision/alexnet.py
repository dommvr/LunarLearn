from LunarLearn.nn import Module
from LunarLearn.nn.layers import BaseLayer, BatchNorm2D, Conv2D, MaxPool2D, Flatten, Dense
from LunarLearn.core import Tensor


class ConvBlock(BaseLayer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 groups=1,
                 use_norm=True,
                 activation="relu",
                 use_pool=True):
        from LunarLearn.nn.activations import get_activation
        super().__init__(trainable=True)
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           groups=groups)
        self.norm = BatchNorm2D() if use_norm else None
        self.activation = get_activation(activation)
        self.pool = MaxPool2D(pool_size=3, strides=2) if use_pool else None

    def forward(self, x: Tensor):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class AlexNet(Module):
    def __init__(self,
                 num_classes=1000,
                 use_norm=False,
                 dropout=0.5,
                 activation="relu",
                 final_activation=None,
                 pretrained=False):  # e.g. "softmax" or None for logits
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        self.block1 = ConvBlock(filters=96,
                                kernel_size=11,
                                strides=4,
                                padding=2,
                                use_norm=use_norm,
                                activation=activation,
                                use_pool=True)
        
        self.block2 = ConvBlock(filters=256,
                                kernel_size=5,
                                strides=1,
                                padding=2,
                                groups=2,
                                use_norm=use_norm,
                                activation=activation,
                                use_pool=True)
        
        self.block3 = ConvBlock(filters=384,
                                kernel_size=3,
                                strides=1,
                                padding=1,
                                use_norm=use_norm,
                                activation=activation,
                                use_pool=False)
        
        self.block4 = ConvBlock(filters=384,
                                kernel_size=3,
                                strides=1,
                                padding=1,
                                groups=2,
                                use_norm=use_norm,
                                activation=activation,
                                use_pool=False)
        
        self.block5 = ConvBlock(filters=256,
                                kernel_size=3,
                                strides=1,
                                padding=1,
                                groups=2,
                                use_norm=use_norm,
                                activation=activation,
                                use_pool=True)
        
        self.flatten = Flatten()

        self.fc1 = Dense(nodes=4096,
                         activation=activation,
                         dropout=dropout)
        
        self.fc2 = Dense(nodes=4096,
                         activation=activation,
                         dropout=dropout)
        
        self.fc3 = Dense(num_classes)
        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.final_act(x)
        return out