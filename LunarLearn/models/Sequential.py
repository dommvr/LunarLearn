import LunarLearn.backend as backend
from LunarLearn.models.Module import Module 
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

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
