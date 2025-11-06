from LunarLearn.nn.layers import BaseLayer
from LunarLearn.nn import Sequential
from LunarLearn.core import Tensor
from LunarLearn.core.tensor import checkpoint

class CheckpointBlock(BaseLayer):
    """
    Wrap any sub-layer or callable block to checkpoint its forward.

    Example:
        block = Sequential(Conv2D(...), BatchNorm2D(...), ReLU())
        x = CheckpointBlock(block)(x)
    """
    def __init__(self, block):
        super().__init__(trainable=True)  # params live inside block
        if isinstance(block, list):
            block = Sequential(*block)
        self.block = block

    def initialize(self, input_shape):
        # delegate to sub-block if it requires shape lazily
        if hasattr(self.block, "initialize"):
            self.block.initialize(input_shape)
        self.output_shape = getattr(self.block, "output_shape", None)

    def forward(self, x: Tensor) -> Tensor:
        # Important: pass a closure that uses current training flag
        def fn(inp):
            # Make sure the sub-block sees the current training mode if it uses it
            if hasattr(self.block, "training"):
                self.block.training = self.training
            return self.block(inp)

        return checkpoint(fn, x)
