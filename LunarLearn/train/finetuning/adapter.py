from LunarLearn.nn.layers import BaseLayer, Dense
from LunarLearn.nn.activations import get_activation

class Adapter(BaseLayer):
    def __init__(self, layer, d_model, bootleneck=64):
        super().__init__(trainable=True)
        self.layer = layer
        self.down = Dense(bootleneck)
        self.up = Dense(d_model)
        self.act = get_activation("gelu")

    def forward(self, x):
        return self.layer(x) + self.up(self.act(self.down(x)))