from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import ConvLSTM, GlobalAveragePool2D, Dropout, Dense
from LunarLearn.core import Tensor


class ConvLSTMClassifier(Module):
    """
    Simple video classifier:
    stacked ConvLSTM -> (take last frame) -> GlobalAveragePool2D -> Dense(num_classes)
    """
    def __init__(self,
                 hidden_channels_list: list[int],
                 num_classes: int,
                 kernel_size: int | list | tuple = 3,
                 padding: int | str = "same",
                 keep_prob: float = 0.8,
                 return_sequences: bool = False,
                 final_activation: str | None = None):
        from LunarLearn.nn.activations import get_activation
        super().__init__()

        self.return_sequences = return_sequences

        # Normalize kernel_size to per-layer list if needed
        if isinstance(kernel_size, (int, tuple)):
            kernel_sizes = [kernel_size] * len(hidden_channels_list)
        else:
            kernel_sizes = list(kernel_size)
            assert len(kernel_sizes) == len(hidden_channels_list)

        # Stack ConvLSTM layers, all returning sequences (B, T, C, H, W)
        self.convlstm_block = ModuleList([
            ConvLSTM(
                hidden_channels=c,
                kernel_size=k,
                padding=padding,
                return_sequences=True    # <- force sequence output inside stack
            )
            for c, k in zip(hidden_channels_list, kernel_sizes)
        ])

        self.global_pool = GlobalAveragePool2D()
        self.dropout = Dropout(keep_prob=keep_prob)
        self.fc = Dense(num_classes)

        self.final_act = get_activation(final_activation)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, C, H, W)
        """
        out = self.convlstm_block(x)

        # Decide what to do with time dimension
        if not self.return_sequences:
            # take last time step
            out = out[:, -1, :, :, :]    # (B, C_last, H, W)
        else:
            # or you could pool over time here if you wanted, but let's keep shape
            # e.g. out = ops.mean(out, axis=1)
            out = out[:, -1, :, :, :]    # simple choice: still last step

        out = self.global_pool(out)      # (B, C_last)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.final_act(out)
        return out