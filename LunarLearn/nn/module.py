import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Module:
    """
    Base class for models, similar to torch.nn.Module.

    - Subclass this to create custom models.
    - Define layers as attributes in __init__.
    - Implement forward() to define computation.
    - Parameters are automatically collected from child layers.
    """
    def __init__(self):
        self.training = True

    def __call__(self, *inputs, **kwargs) -> Tensor:
        return self.forward(*inputs, **kwargs)
    
    def __repr__(self):
        class_name = self.__class__.__name__
        
        # Sequential-style model (list of layers)
        if hasattr(self, "layers") and isinstance(self.layers, (list, tuple)):
            child_strs = []
            total_params = 0
            for i, layer in enumerate(self.layers):
                layer_str = repr(layer).replace("\n", "\n    ")
                child_strs.append(f"  ({i}): {layer_str}")
                total_params += layer.count_parameters() if hasattr(layer, "count_parameters") else 0

            child_str = "\n".join(child_strs)
            return f"{class_name}(\n{child_str}\n)\nTotal params: {total_params}"
        
        else:
            # Single layer
            extra = self.extra_repr()
            params = self.count_parameters()
            if extra:
                return f"{class_name}({extra}, params={params})"
            else:
                return f"{class_name}(params={params})"

    def parameters(self, with_layer: bool = False):
        """
        Collect all trainable parameters.

        Args:
            with_layer (bool, optional): 
                - If False (default), return a flat list of Tensors. 
                - If True, return a list of dicts {"param": Tensor, "layer": BaseLayer}
                for optimizers that need layer-level context.

        Returns:
            List[Tensor | dict]: List of parameters (with or without layer info).
        """
        return [p for _, p in self.named_parameters(with_layer=with_layer)]

    def named_parameters(self, prefix: str = "", with_layer: bool = False):
        """
        Recursively collect all trainable parameters with names.

        Args:
            prefix (str, optional): Name prefix for recursion (used internally).
            with_layer (bool, optional): 
                - If False, return (name, Tensor) pairs.
                - If True, return (name, {"param": Tensor, "layer": BaseLayer}) pairs.

        Returns:
            List[Tuple[str, Tensor | dict]]: Named parameters.
        """
        params = []
        for name, attr in self.__dict__.items():
            if isinstance(attr, BaseLayer) and not attr.trainable:
                continue
            if isinstance(attr, BaseLayer):
                params += attr.named_parameters(prefix=f"{prefix}{name}.", with_layer=with_layer)
            elif isinstance(attr, Module):
                params += attr.named_parameters(prefix=f"{prefix}{name}.", with_layer=with_layer)
        return params
    
    def layers(self):
        for name, module in self.__dict__.items():
            if isinstance(module, BaseLayer):
                yield module

    def forward(self, *inputs, **kwargs) -> Tensor:
        raise NotImplementedError("Subclasses must implement forward().")

    def train(self):
        """Set the module and all children to training mode."""
        self.training = True
        for attr in self.__dict__.values():
            if hasattr(attr, "train"):
                attr.train()

    def eval(self):
        """Set the module and all children to evaluation mode."""
        self.training = False
        for attr in self.__dict__.values():
            if hasattr(attr, "eval"):
                attr.eval()

    def freeze(self):
        """Freeze all parameters in this module."""
        for p in self.parameters():
            p.frozen = True

    def unfreeze(self):
        """Unfreeze all parameters in this module."""
        for p in self.parameters():
            p.frozen = False

    def summary(self, input_shape: tuple = None, as_dict: bool = False):
        """
        Print or return a detailed summary of the model, similar to Keras/PyTorch.

        Args:
            input_shape (tuple, optional): Shape of the input tensor 
                excluding batch size. If provided, a dummy forward pass 
                is performed to infer output shapes.
            as_dict (bool, optional): If True, return summary as a 
                dictionary instead of printing. Default is False.

        Returns:
            dict, optional: If as_dict=True, returns a dictionary with keys:
                - "layers": list of dicts, each with:
                    - "name": str
                    - "output_shape": tuple or str
                    - "params": int
                    - "trainable": bool
                - "total_params": int
                - "trainable_params": int
                - "non_trainable_params": int
                - "memory_MB": float
        """
        import numpy as np

        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        memory_bytes = 0
        layers_info = []

        x = None
        if input_shape is not None:
            dummy = np.zeros((1, *input_shape), dtype="float32")
            x = Tensor(dummy)

        for i, (name, layer) in enumerate(self._modules.items()):
            layer_name = f"({i}) {layer.__class__.__name__}"
            params = 0
            dtype_size = 4  # default float32, updated below

            # Count params if layer has weights/bias
            if hasattr(layer, "count_parameters"):
                params = layer.count_parameters()

                # try to infer dtype size from weights if present
                if hasattr(layer, "W") and layer.W is not None:
                    dtype_size = np.dtype(layer.W.dtype).itemsize

            total_params += params
            memory_bytes += params * dtype_size

            if getattr(layer, "trainable", False):
                trainable_params += params
            else:
                non_trainable_params += params

            out_shape = "-"
            if x is not None:
                try:
                    x = layer(x)
                    out_shape = tuple(x.shape)
                except Exception:
                    out_shape = "?"

            layers_info.append({
                "name": layer_name,
                "output_shape": out_shape,
                "params": params,
                "trainable": getattr(layer, "trainable", False),
            })

        memory_MB = memory_bytes / (1024 ** 2)

        if as_dict:
            return {
                "layers": layers_info,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "non_trainable_params": non_trainable_params,
                "memory_MB": round(memory_MB, 2),
            }

        # Pretty print summary
        print("=" * 90)
        print(f"{self.__class__.__name__} Summary")
        print("=" * 90)
        print(f"{'Layer':25} {'Output Shape':25} {'Param #':15}")
        print("-" * 90)
        for layer in layers_info:
            print(f"{layer['name']:25} {str(layer['output_shape']):25} {layer['params']:15,d}")
        print("=" * 90)
        print(f"Total params:        {total_params:,}")
        print(f"Trainable params:    {trainable_params:,}")
        print(f"Non-trainable params:{non_trainable_params:,}")
        print(f"Model size (params): {memory_MB:.2f} MB")
        print("=" * 90)