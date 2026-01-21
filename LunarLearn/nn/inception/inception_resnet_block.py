from LunarLearn.nn import ModuleList
from LunarLearn.nn.inception import Inception
from LunarLearn.nn.layers import BatchNorm2D, Conv2D
from LunarLearn.core import Tensor


class InceptionResNetBlock(Inception):
    def __init__(
        self,
        f_1x1,
        f_3x3_reduce,
        f_3x3,
        f_5x5_reduce,
        f_5x5,
        scale=0.1,
        norm_layer=BatchNorm2D,
        activation="relu"
    ):
        self.norm_layer = norm_layer
        self.activation = activation
        self.scale = scale

        branches = []

        # Branch 1: 1×1 conv
        branches.append(
            ModuleList(self._make_conv_layers([
                Conv2D(f_1x1, kernel_size=1, padding="same")
            ]))
        )

        # Branch 2: 1×1 -> 3×3
        branches.append(
            ModuleList(self._make_conv_layers([
                Conv2D(f_3x3_reduce, kernel_size=1, padding="same"),
                Conv2D(f_3x3, kernel_size=3, padding="same")
            ]))
        )

        # Branch 3: 1×1 -> 5×5
        branches.append(
            ModuleList(self._make_conv_layers([
                Conv2D(f_5x5_reduce, kernel_size=1, padding="same"),
                Conv2D(f_5x5, kernel_size=5, padding="same")
            ]))
        )

        super().__init__(branches, norm_layer=norm_layer, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        from LunarLearn.nn.activations import get_activation
        out = super().forward(x)
        # Residual scaling to stabilize training
        out = x + self.scale * out

        activation = get_activation(self.activation)
        out = activation(out)
        return out