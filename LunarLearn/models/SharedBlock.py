from LunarLearn.layers.BaseLayer import BaseLayer

class SharedBlock(BaseLayer):
    def __init__(self, block):
        super().__init__(trainable=block.trainable)
        self.block = block

    def forward(self, x, **kwargs):
        return self.block(x, **kwargs)