from LunarLearn.nn.layers import BaseLayer, Dense, ReLU, Conv2D, BatchNorm2D, GlobalAveragePool2D
from LunarLearn.core import ops
from LunarLearn.nn import Module, ModuleList

class SEBlock(BaseLayer):
    def __init__(self, channels, reduction=4):
        super().__init__(trainable=True)
        self.fc1 = Dense(channels // reduction)
        self.fc2 = Dense(channels)
        self.activation = ReLU()

    def forward(self, x):
        w = ops.mean(x, axis=[2, 3], keepdims=True)
        w = w.reshape(w.shape[0], -1)
        w = self.fc1(w)
        w = self.activation(w)
        w = self.fc2(w)
        w = ops.sigmoid(w)
        w = w.reshape(-1, w.shape[1], 1, 1)
        out = x * w
        return out
    

class RegNetBlock(BaseLayer):
    def __init__(self, out_channels, strides=1, expansion=1):
        super().__init__(trainable=True)
        mid_channels = out_channels // expansion
        self.conv1 = Conv2D(mid_channels, kernel_size=1)
        self.norm1 = BatchNorm2D()
        self.conv2 = Conv2D(mid_channels, kernel_size=3, strides=strides, padding=1)
        self.norm2 = BatchNorm2D()
        self.conv3 = Conv2D(out_channels, kernel_size=1)
        self.norm3 = BatchNorm2D()
        self.se = SEBlock(out_channels)
        self.activation = ReLU()

        self.shortcut = None
        
    def _make_shortcut(self, input_shape):
        in_channels = input_shape[1]
        out_channels = self.filters

        if in_channels != out_channels or self.strides != 1:
            self.shortcut = ModuleList([Conv2D(filters=out_channels,
                                        kernel_size=1,
                                        strides=self.strides),
                                        BatchNorm2D()])
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        if self.shortcut is None:
            self._make_shortcut(x.shape)
        
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.se(x)
        out = x + identity
        out = self.activation(out)
        return out
    

class RegNetY16GF(Module):
    def __init__(self, n_classes=1000, pretrained=False):
        super().__init__()
        # Config from paper: depths=[2, 6, 17, 2], widths=[232, 464, 928, 2080]
        depths = [2, 6, 17, 2]
        widths = [232, 464, 928, 2080]
        strides = [2, 2, 2, 2]
        expansion = 4  # RegNetY uses group width = w_m / g = 232 / 4 = 58

        # Stem
        self.stem = ModuleList([
            Conv2D(32, kernel_size=3, strides=2, padding=1),
            BatchNorm2D(),
            ReLU()
        ])

        self.trunk = ModuleList()
        for i in range(4):
            stage_layers = ModuleList()
            out_channels = widths[i]
            for j in range(depths[i]):
                strides = strides[i] if j == 0 else 1
                stage_layers.append(RegNetBlock(out_channels, strides=strides, expansion=expansion))
            self.trunk.append(stage_layers)

        self.head = ModuleList([GlobalAveragePool2D(),
                                Dense(n_classes)])
        
        if pretrained:
            self.load_state_dict(None)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        out = self.head(x)
        return out