from LunarLearn.layers.RecurrentBase import RecurrentBase
from LunarLearn.layers.GRUCell import GRUCell

class GRU(RecurrentBase):
    """
    Gated Recurrent Unit (GRU) layer.

    Extends `RecurrentBase` to provide a full GRU implementation with:
        - Stateful hidden states across batches
        - Recurrent dropout between timesteps
        - Mixed precision and GPU compatibility
        - Autograd-friendly Tensor operations

    This layer unrolls a `GRUCell` over input timesteps, managing hidden state,
    dropout, and dtype consistency automatically. It supports both returning
    the full output sequence or only the last hidden state.

    Args:
        hidden_size (int): Number of hidden units in the GRU cell.
        return_sequences (bool, optional): If True, returns the full sequence of
            hidden states (shape: `(batch, timesteps, hidden_size)`).
            If False, returns only the last hidden state (shape: `(batch, hidden_size)`).
            Defaults to False.
        activation (str, optional): Activation function used in candidate hidden state.
            Typically `"tanh"`. Defaults to `"tanh"`.
        w_init (str, optional): Weight initialization strategy (e.g. `"xavier"`, `"he"`, `"uniform"`).
            Defaults to `"auto"`.
        uniform (bool, optional): Whether to sample weights from a uniform distribution.
            Defaults to False.
        gain (float, optional): Optional scaling gain applied during initialization.
            Defaults to 1.
        stateful (bool, optional): If True, retains hidden states across batches.
            Useful for sequence continuation tasks. Defaults to False.
        recurrent_keep_prob (float, optional): Dropout keep probability applied to
            the recurrent connection (`h_{t-1}`). Must be in (0, 1].
            Defaults to 1.0 (no dropout).

    Returns:
        Tensor: Output tensor of shape:
            - `(batch, timesteps, hidden_size)` if `return_sequences=True`
            - `(batch, hidden_size)` otherwise
    """
    def __init__(self, hidden_size, return_sequences=False,
                 activation="tanh", w_init="auto", uniform=False, gain=1,
                 stateful=False, recurrent_keep_prob=1.0):
        super().__init__(GRUCell, hidden_size, return_sequences,
                         activation, w_init, uniform, gain,
                         stateful=stateful, recurrent_keep_prob=recurrent_keep_prob)