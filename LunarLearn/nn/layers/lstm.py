from LunarLearn.nn.layers import RecurrentBase
from LunarLearn.nn.layers import LSTMCell

class LSTM(RecurrentBase):
    """
    Long Short-Term Memory (LSTM) layer.

    Extends `RecurrentBase` to manage both hidden and cell states using an `LSTMCell`.
    Supports stateful sequence processing, recurrent dropout, and mixed precision.

    Args:
        hidden_size (int): Number of hidden units.
        return_sequences (bool, optional): Whether to return all timesteps or only the last.
            Defaults to False.
        activation (str, optional): Activation function for the candidate cell state.
            Defaults to `"tanh"`.
        w_init (str, optional): Weight initialization strategy. Defaults to `"auto"`.
        uniform (bool, optional): Use uniform initialization. Defaults to False.
        gain (float, optional): Initialization gain. Defaults to 1.
        stateful (bool, optional): Retain hidden and cell states across batches.
            Defaults to False.
        recurrent_keep_prob (float, optional): Dropout keep probability for recurrent state.
            Defaults to 1.0.
    """
    def __init__(self, hidden_size, return_sequences=False,
                 activation="tanh", w_init="auto", uniform=False, gain=1,
                 stateful=False, recurrent_keep_prob=1.0):
        super().__init__(LSTMCell, hidden_size, return_sequences,
                         activation, w_init, uniform, gain,
                         stateful=stateful, recurrent_keep_prob=recurrent_keep_prob)