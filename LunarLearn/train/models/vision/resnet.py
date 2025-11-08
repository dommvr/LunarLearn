from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import Conv2D, BatchNorm2D, ReLU, MaxPool2D, GlobalAveragePool2D, Dense, Flatten
from LunarLearn.nn.resblocks import BasicResBlock, BottleneckResBlock, ResNeXtBlock
from LunarLearn.core import Tensor

class ResNet(Module):
    def __init__(self, block, layers, num_classes, stem, groups=None, pretrained=False):
        super().__init__()
        self.stem = ModuleList(Conv2D(64, kernel_size=7, strides=2, padding=3),
                               BatchNorm2D(),
                               ReLU(),
                               MaxPool2D(pool_size=3, strides=2, padding=1)) if stem else None
        
        self.layer1 = self._make_layer(block, layers[0], 64, strides=1)
        self.layer2 = self._make_layer(block, layers[1], 128, strides=2)
        self.layer3 = self._make_layer(block, layers[2], 256, strides=2)
        self.layer4 = self._make_layer(block, layers[3], 512, strides=2)

        self.head = ModuleList(GlobalAveragePool2D(),
                               Flatten(),
                               Dense(num_classes))
        
        self.groups = groups

        self._zero_init_residual()

        if pretrained:
            self.load_state_dict(None) #will add weights layter

    def _make_layer(self, block, num_blocks, filters, strides):
        if self.groups is not None:
            blocks = [block(filters=filters, strides=strides if n == 0 else 1, groups=self.groups) for n in range(num_blocks)]
        else:
            blocks = [block(filters=filters, strides=strides if n == 0 else 1) for n in range(num_blocks)]
        return ModuleList(blocks)
    
    def _zero_init_residual(self):
        for m in self.modules():
            if hasattr(m, 'norm2'):  # BasicBlock
                m.norm2.zero_init()
            elif hasattr(m, 'norm3'):  # Bottleneck
                m.norm3.zero_init()
    
    def forward(self, x: Tensor) -> Tensor:
        if self.stem is not None:
            x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        out = self.head(x)
        return out
    

class ResNet18(ResNet):
    def __init__(self, num_classes=1000, stem=True, pretrained=False):
        super().__init__(BasicResBlock, [2, 2, 2, 2], num_classes=num_classes, stem=stem, pretrained=pretrained)


class ResNet34(ResNet):
    def __init__(self, num_classes=1000, stem=True, pretrained=False):
        super().__init__(BasicResBlock, [3, 4, 6, 3], num_classes=num_classes, stem=stem, pretrained=pretrained)


class ResNet50(ResNet):
    def __init__(self, num_classes=1000, stem=True, pretrained=False):
        super().__init__(BottleneckResBlock, [3, 4, 6, 3], num_classes=num_classes, stem=stem, pretrained=pretrained)


class ResNeXt50(ResNet):
    def __init__(self, num_classes=1000, stem=True, groups=32, pretrained=False):
        super().__init__(ResNeXtBlock, [3, 4, 6, 3], num_classes=num_classes, stem=stem, groups=groups, pretrained=pretrained)