from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, Conv2D, MaxPool2D, Activation, BatchNorm2D, Flatten, Dense
from LunarLearn.core import Tensor


_VGG_CONFIGS = {
    # VGG11 (A)
    "VGG11": [64, "M",
              128, "M",
              256, 256, "M",
              512, 512, "M",
              512, 512, "M"],

    # VGG13 (B)
    "VGG13": [64, 64, "M",
              128, 128, "M",
              256, 256, "M",
              512, 512, "M",
              512, 512, "M"],

    # VGG16 (D)
    "VGG16": [64, 64, "M",
              128, 128, "M",
              256, 256, 256, "M",
              512, 512, 512, "M",
              512, 512, 512, "M"],

    # VGG19 (E)
    "VGG19": [64, 64, "M",
              128, 128, "M",
              256, 256, 256, 256, "M",
              512, 512, 512, 512, "M",
              512, 512, 512, 512, "M"],
}


class VGGBlock(BaseLayer):
    def __init__(self,
                 out_channels,
                 num_convs,
                 use_norm=False,
                 activation="relu"):
        super().__init__(trainable=True)

        self.layers = ModuleList()

        for _ in range(num_convs):
            conv = Conv2D(filters=out_channels,
                          kernel_size=3,
                          strides=1,
                          padding=1,
                          bias=not use_norm)
            self.layers.append(conv)
            if use_norm:
                self.layers.append(BatchNorm2D())
            self.layers.append(Activation(activation))

        self.pool = MaxPool2D(pool_size=2, strides=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.pool(x)
        return x
    

class VGG(Module):
    def __init__(self,
                 arch="VGG16",
                 num_classes=1000,
                 use_norm=False,
                 fc_hidden=4096,
                 keep_prob=0.5,
                 activation="relu",
                 final_activation=None,
                 pretrained=False):
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        if arch not in _VGG_CONFIGS:
            raise ValueError(f"Unknown VGG architecture '{arch}'. "
                             f"Expected one of: {list(_VGG_CONFIGS.keys())}")

        cfg = _VGG_CONFIGS[arch]

        # ----- Features (conv blocks + pooling) -----
        self.features = self._make_features(cfg,
                                            use_norm=use_norm,
                                            activation=activation)

        self.flatten = Flatten()

        # ----- Classifier -----
        # Classic VGG: 4096 -> 4096 -> num_classes
        # fc_hidden is a parameter in case you want to shrink it
        self.fc1 = Dense(nodes=fc_hidden,
                         activation=activation,
                         keep_prob=keep_prob)
        self.fc2 = Dense(nodes=fc_hidden,
                         activation=activation,
                         keep_prob=keep_prob)
        self.fc3 = Dense(nodes=num_classes)

        self.final_act = get_activation(final_activation)

        if pretrained:
            self.load_state_dict(None)

    def _make_features(self, cfg, use_norm, activation):
        layers = ModuleList()

        i = 0
        while i < len(cfg):
            if cfg[i] == "M":
                # This should never be first, but just in case
                layers.append(MaxPool2D(pool_size=2, strides=2))
                i += 1
                continue

            # count how many consecutive convs with same out_channels before next "M"
            out_channels = cfg[i]
            num_convs = 1
            j = i + 1
            while j < len(cfg) and cfg[j] == out_channels:
                num_convs += 1
                j += 1
            # if next is "M", block ends there
            # build a block with num_convs convolutions then a pool
            block = VGGBlock(out_channels=out_channels,
                             num_convs=num_convs,
                             use_norm=use_norm,
                             activation=activation)
            layers.append(block)

            # skip the conv entries we just consumed
            # and skip the "M" that follows (pool is inside the block)
            i = j
            if i < len(cfg) and cfg[i] == "M":
                i += 1

        return layers
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.final_act(x)
        return out
    

class VGG11(VGG):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(arch="VGG11",
                         num_classes=num_classes,
                         pretrained=pretrained,
                         **kwargs)


class VGG13(VGG):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(arch="VGG13",
                         num_classes=num_classes,
                         pretrained=pretrained,
                         **kwargs)


class VGG16(VGG):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(arch="VGG16",
                         num_classes=num_classes,
                         pretrained=pretrained,
                         **kwargs)


class VGG19(VGG):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super().__init__(arch="VGG19",
                         num_classes=num_classes,
                         pretrained=pretrained,
                         **kwargs)