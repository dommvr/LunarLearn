from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.layers import Dense, Activation, Dropout
from LunarLearn.tensor.tensor import Tensor

class FeedForward(BaseLayer):
    def __init__(self, layer1_nodes, layer2_nodes, activation="relu", keep_prob=0.9):
        super().__init__(trainable=True)
        self.dense1 = Dense(nodes=layer1_nodes)
        self.dense2 = Dense(nodes=layer2_nodes)
        self.activation = Activation(activation)
        self.dropout = Dropout(keep_prob)

    def forward(self, x: Tensor) -> Tensor:
        out = self.dense1(x)
        out = self.activation(out)
        out = self.dense2(out)
        out = self.dropout(out)

        return out