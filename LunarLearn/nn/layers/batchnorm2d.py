from LunarLearn.nn.layers import BatchNorm

class BatchNorm2D(BatchNorm):
    def __init__(self, momentum=0.9, epsilon=0.001):
        super().__init__(ndim=2, momentum=momentum, epsilon=epsilon)