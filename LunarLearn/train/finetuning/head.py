from LunarLearn.nn.layers import BaseLayer

class Head(BaseLayer):
    def __init__(self):
        super().__init__(trainable=True)

def add_head(model):
    pass