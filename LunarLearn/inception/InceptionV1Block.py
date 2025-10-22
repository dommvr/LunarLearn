from LunarLearn.inception.Inception import Inception
from LunarLearn.inception import InceptionBranch
from LunarLearn.layers import Conv2D, MaxPool2D, AveragePool2D, BatchNorm2D

class InceptionV1Block(Inception):
    def __init__(
        self,
        f_1x1: int,
        f_3x3_reduce: int, f_3x3: int,
        f_5x5_reduce: int, f_5x5: int,
        f_pool_proj: int,
        pool_type: str = "max",
        norm_layer=BatchNorm2D,
        activation: str = "relu",
    ):
        self.norm_layer = norm_layer
        self.activation = activation

        branches = []

        # Branch 1: 1x1 conv
        branches.append(InceptionBranch(self._make_conv_layers([
            Conv2D(filters=f_1x1, kernel_size=1, padding="same")
        ])))

        # Branch 2: 1x1 -> 3x3
        branches.append(InceptionBranch(self._make_conv_layers([
            Conv2D(filters=f_3x3_reduce, kernel_size=1, padding="same"),
            Conv2D(filters=f_3x3, kernel_size=3, padding="same")
        ])))

        # Branch 3: 1x1 -> 5x5
        branches.append(InceptionBranch(self._make_conv_layers([
            Conv2D(filters=f_5x5_reduce, kernel_size=1, padding="same"),
            Conv2D(filters=f_5x5, kernel_size=5, padding="same")
        ])))

        # Branch 4: Pool -> 1x1
        pool_layer = MaxPool2D(pool_size=3, strides=1, padding="same") if pool_type == "max" \
                    else AveragePool2D(pool_size=3, strides=1, padding="same")
        pool_branch_layers = [pool_layer]
        pool_branch_layers.extend(self._make_conv_layers([
            Conv2D(filters=f_pool_proj, kernel_size=1, padding="same")
        ]))
        branches.append(InceptionBranch(pool_branch_layers))

        super().__init__(branches)