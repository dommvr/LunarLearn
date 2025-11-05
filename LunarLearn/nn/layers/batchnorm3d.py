from LunarLearn.layers.BatchNorm import BatchNorm

class BatchNorm3D(BatchNorm):
    def __init__(self, momentum=0.9, epsilon=0.001):
        super().__init__(ndim=3, momentum=momentum, epsilon=epsilon)