import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp

DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

class RecurrentBase(BaseLayer):
    """
    Base class for all recurrent layers (RNN, GRU, LSTM).

    Handles sequence unrolling, mixed precision, stateful behavior,
    and recurrent dropout. Subclasses define their specific recurrent
    cell type (e.g., `RNNCell`, `GRUCell`, `LSTMCell`).

    Features:
        - Stateful operation (retain hidden state between batches)
        - Recurrent dropout with per-sequence masking
        - Autograd-compatible sequence unrolling
        - Mixed precision support

    Args:
        cell_class (type): The recurrent cell class to unroll.
        hidden_size (int): Number of hidden units.
        return_sequences (bool, optional): If True, returns all timesteps.
            Defaults to False.
        activation (str, optional): Activation function used in cell.
            Defaults to `"tanh"`.
        w_init (str, optional): Weight initialization method. Defaults to `"auto"`.
        uniform (bool, optional): Use uniform distribution for initialization.
            Defaults to False.
        gain (float, optional): Scaling gain for initialization. Defaults to 1.
        stateful (bool, optional): Retain hidden state across batches.
            Defaults to False.
        recurrent_keep_prob (float, optional): Dropout keep probability on hidden state.
            Defaults to 1.0.

    Returns:
        Tensor: Output of shape `(batch, timesteps, hidden_size)` or `(batch, hidden_size)`.
    """
    def __init__(self, cell_class, hidden_size, return_sequences=False,
                 activation="tanh", w_init="auto", uniform=False, gain=1,
                 stateful=False, recurrent_keep_prob=1.0):
        super().__init__(trainable=True)
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        self.cell_class = cell_class
        self.cell = cell_class(hidden_size, activation, w_init, uniform, gain)

        # Stateful and dropout options
        self.stateful = stateful
        self.recurrent_keep_prob = recurrent_keep_prob
        self.h_state = None  # persistent hidden state if stateful

        # Caches
        self.h_cache = None
        self.x_cache = None

    def reset_state(self, batch_size=None):
        """Reset internal state (used between epochs or sequences)."""
        self.h_state = None if batch_size is None else Tensor(
            xp.zeros((batch_size, self.hidden_size), dtype=DTYPE), requires_grad=True
        )

    def initialize(self, input_shape):
        timesteps, features = input_shape
        self.cell.initialize((features,))
        self.output_shape = (
            (timesteps, self.hidden_size)
            if self.return_sequences else
            (self.hidden_size,)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through recurrent sequence with dropout, mixed precision, and stateful control.

        Args:
            inputs (Tensor): (batch, timesteps, features)
            training (bool): Whether in training mode

        Returns:
            Tensor: Output (batch, hidden_size) or (batch, timesteps, hidden_size)
        """
        from LunarLearn.regularizers import RecurrentDropout

        batch_size, timesteps, features = inputs.shape

        # Initialize or reuse hidden state
        h = (
            self.h_state
            if self.stateful and self.h_state is not None
            else Tensor(xp.zeros((batch_size, self.hidden_size), dtype=DTYPE), requires_grad=True)
        )

        hs = []
        self.x_cache, self.h_cache = inputs, []

        # Pre-create dropout mask once per sequence
        dropout = RecurrentDropout((batch_size, self.hidden_size), self.recurrent_keep_prob, self.training)

        for t in range(timesteps):
            x_t = inputs[:, t, :]

            # Recurrent Dropout
            h = dropout(h)

            # Forward step
            h = self.cell.forward(x_t, h)

            h_cache_entry = h.astype(DTYPE, copy=False)
            self.h_cache.append(h_cache_entry)
            if self.return_sequences:
                hs.append(h_cache_entry)

        # Stateful Handling
        if self.stateful:
            self.h_state = h.detach()  # store FP32 master hidden state

        # Output Formatting
        if self.return_sequences:
            out = ops.stack(hs, axis=1)  # (batch, timesteps, hidden)
        else:
            out = h

        return out