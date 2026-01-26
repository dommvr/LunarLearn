import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import RecurrentBase
from LunarLearn.nn.layers import RNNCell

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class RNN(RecurrentBase):
    """
    Vanilla RNN layer.

    Sequentially applies an `RNNCell` across timesteps to process sequential data.
    Supports:
        - Stateful hidden states
        - Recurrent dropout
        - Mixed precision

    Args:
        hidden_size (int): Number of hidden units.
        return_sequences (bool, optional): Return full sequence or only last state.
            Defaults to False.
        activation (str, optional): Activation function (default: `"tanh"`).
        w_init (str, optional): Weight initialization method. Defaults to `"auto"`.
        uniform (bool, optional): Use uniform initialization. Defaults to False.
        gain (float, optional): Scaling gain. Defaults to 1.
        stateful (bool, optional): Retain hidden states across batches. Defaults to False.
        recurrent_keep_prob (float, optional): Dropout keep probability for hidden state.
            Defaults to 1.0.
    """
    def __init__(self, hidden_size, return_sequences=False, activation="tanh",
                 w_init="auto", uniform=False, gain=1,
                 stateful=False, recurrent_keep_prob=1.0, zero_bias=True, bias=True):
        super().__init__(RNNCell, hidden_size, return_sequences,
                         activation, w_init, uniform, gain,
                         stateful=stateful, recurrent_keep_prob=recurrent_keep_prob,
                         zero_bias=zero_bias, bias=bias)