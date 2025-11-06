import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class Dropout(BaseLayer):
    """
    Dropout layer with autograd support.

    Randomly drops activations during training with probability 
    (1 - keep_prob) and rescales the remaining values. This helps 
    prevent overfitting. In inference mode, dropout is bypassed.

    Args:
        keep_prob (float, optional):
            Probability of keeping a unit active (0 < keep_prob < 1).
            Default is 0.8.

    Attributes:
        keep_prob (float):
            Probability of retaining each activation.
        inv_keep_prob (float):
            Scaling factor applied during training (1 / keep_prob).
        output_shape (tuple):
            Same as input shape, set during initialization.

    Methods:
        initialize(input_shape):
            Sets the output shape to match the input shape.
        forward(A_prev: Tensor) -> Tensor:
            Applies dropout during training, or passes input unchanged
            during inference.
    """
    def __init__(self, keep_prob=0.8):

        # Validate keep_prob
        if not isinstance(keep_prob, (int, float)):
            raise ValueError("keep_prob must be a float or 1")
        if not (0 < keep_prob < 1):
            raise ValueError("keep_prob must be in the range (0, 1)")
        
        super().__init__(trainable=False)

        self.keep_prob = xp.array(keep_prob, dtype=DTYPE)
        
    def initialize(self, input_shape):
        if input_shape is None:
            raise ValueError("input_shape must be provided to initialize the layer")
    
        self.output_shape = input_shape

    def forward(self, A_prev: Tensor) -> Tensor:
        return ops.dropout(A_prev, self.keep_prob, training=self.training)