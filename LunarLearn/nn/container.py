class ModuleList(BaseLayer):
    def __init__(self, modules=None):
        super().__init__(trainable=True)
        # store list of submodules (EncoderBlock, DecoderBlock, etc.)
        self.modules = list(modules) if modules is not None else []

    def append(self, module):
        """Add a new submodule to the list."""
        self.modules.append(module)

    def extend(self, modules):
        """Add multiple submodules."""
        self.modules.extend(modules)

    def __getitem__(self, idx):
        return self.modules[idx]

    def __len__(self):
        return len(self.modules)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Sequentially pass x through all modules."""
        for m in self.modules:
            x = m(x, *args, **kwargs)
        return x

    def parameters(self, with_layer=False):
        """Collect parameters recursively from all submodules."""
        params = []
        for m in self.modules:
            params.extend(m.parameters(with_layer=with_layer))
        return params

    def named_parameters(self, prefix: str = "", with_layer: bool = False):
        """Same as parameters(), but with names included."""
        params = []
        for i, m in enumerate(self.modules):
            sub_prefix = f"{prefix}block{i}."
            params.extend(m.named_parameters(prefix=sub_prefix, with_layer=with_layer))
        return params
    

class SharedBlock(BaseLayer):
    def __init__(self, block):
        super().__init__(trainable=block.trainable)
        self.block = block

    def forward(self, x, **kwargs):
        return self.block(x, **kwargs)


class Sequential(Module):
    """
    Sequential container for stacking layers.

    Layers are applied in the order they are added.

    Example:
        model = Sequential(
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax")
        )
    """
    def __init__(self, *layers):
        super().__init__()
        self._layers = list(layers)

    def add(self, layer: BaseLayer):
        """Append a new layer to the container."""
        self._layers.append(layer)

    def parameters(self):
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params
    
    def forward(
        self,
        x: Tensor,
        return_activations: bool = False,
        activations_mode: str = "list"  # options: "list", "dict", "both"
    ):
        """
        Forward pass through all layers.

        Args:
            x (Tensor): Input tensor.
            return_activations (bool): Whether to return intermediate activations.
            activations_mode (str): Which activations format to return:
                - "list"  -> returns ordered list of activations
                - "dict"  -> returns dict of activations keyed by layer name
                - "both"  -> returns both list and dict

        Returns:
            - If return_activations=False: Tensor (final output)
            - If return_activations=True:
                - mode="list": (output, activations_list)
                - mode="dict": (output, activations_dict)
                - mode="both": (output, activations_list, activations_dict)
        """
        activations_list = [] if activations_mode in ("list", "both") else None
        activations_dict = {} if activations_mode in ("dict", "both") else None

        for idx, layer in enumerate(self._layers):
            x = layer(x)

            if return_activations:
                if activations_list is not None:
                    activations_list.append(x)
                if activations_dict is not None:
                    name = getattr(layer, "name", None) or layer.__class__.__name__.lower()
                    if name in activations_dict:
                        name = f"{name}_{idx}"
                    activations_dict[name] = x

        if not return_activations:
            return x

        if activations_mode == "list":
            return x, activations_list
        elif activations_mode == "dict":
            return x, activations_dict
        else:  # "both"
            return x, activations_list, activations_dict