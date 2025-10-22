from LunarLearn.inception.Inception import Inception
from LunarLearn.inception import InceptionBranch
from LunarLearn.layers import BatchNorm2D, Conv2D, MaxPool2D, AveragePool2D

class InceptionV2Block(Inception):
    def __init__(
        self,
        f_1x1,
        f_3x3_reduce,
        f_3x3_1,
        f_3x3_2,
        f_5x5_reduce,
        f_5x5_1,
        f_5x5_2,
        f_pool_proj,
        pool_type="max",
        norm_layer=BatchNorm2D,
        activation="relu"
    ):
        self.norm_layer = norm_layer
        self.activation = activation

        branches = []

        # Branch 1: 1×1 conv
        branches.append(
            InceptionBranch(self._make_conv_layers([
                Conv2D(f_1x1, kernel_size=1, padding="same")
            ]))
        )

        # Branch 2: 1×1 -> (1×3) -> (3×1)
        branches.append(
            InceptionBranch(self._make_conv_layers([
                Conv2D(f_3x3_reduce, kernel_size=1, padding="same"),
                Conv2D(f_3x3_1, kernel_size=(1, 3), padding="same"),
                Conv2D(f_3x3_2, kernel_size=(3, 1), padding="same")
            ]))
        )

        # Branch 3: 1×1 -> (1×5) -> (5×1)
        branches.append(
            InceptionBranch(self._make_conv_layers([
                Conv2D(f_5x5_reduce, kernel_size=1, padding="same"),
                Conv2D(f_5x5_1, kernel_size=(1, 5), padding="same"),
                Conv2D(f_5x5_2, kernel_size=(5, 1), padding="same")
            ]))
        )

        # Branch 4: Pool -> 1×1
        pool_layer = MaxPool2D(pool_size=3, strides=1, padding="same") if pool_type == "max" \
                     else AveragePool2D(pool_size=3, strides=1, padding="same")
        pool_branch_layers = [pool_layer]
        pool_branch_layers.extend(self._make_conv_layers([
            Conv2D(f_pool_proj, kernel_size=1, padding="same")
        ]))
        branches.append(InceptionBranch(pool_branch_layers))

        super().__init__(branches, norm_layer=norm_layer, activation=activation)
