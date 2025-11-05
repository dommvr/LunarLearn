from LunarLearn.layers.BatchNorm import BatchNorm

class BatchNorm1D(BatchNorm):
    def __init__(self, momentum=0.9, epsilon=0.001):
        super().__init__(ndim=1, momentum=momentum, epsilon=epsilon)