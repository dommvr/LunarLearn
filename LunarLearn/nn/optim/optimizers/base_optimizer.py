import LunarLearn.backend as backend
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter

xp = backend.xp
DTYPE = backend.DTYPE

class BaseOptimizer:
    """
    Base class for optimizers with autograd support.

    This class defines the interface and common functionality for all optimizers,
    such as learning rate handling, serialization, and integration with schedulers.
    Concrete optimizers (e.g., Adam, AdaBound) should inherit from this class and 
    implement the `step` method.

    Args:
        learning_rate (float): 
            Global learning rate used for parameter updates.

    Attributes:
        learning_rate0 (float): 
            Initial learning rate provided at construction.
        learning_rate (float): 
            Current learning rate (possibly modified by schedulers or scaling).
        scheduler (object, optional): 
            Learning rate scheduler attached to the optimizer.

    Methods:
        get_config() -> dict:
            Serialize optimizer configuration for saving/loading.
        from_config(config: dict) -> BaseOptimizer:
            Restore an optimizer instance from configuration.
        _get_lr(param_desc: Union[Tensor, dict]) -> float:
            Resolve the learning rate for a given parameter or parameter descriptor.
            Supports per-layer overrides and schedulers.
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over parameters.
            Must be implemented by subclasses.
        zero_grad(params: List[Tensor]):
            Set gradients of all given parameters to None.
    """
    def __init__(self, learning_rate):
        self.learning_rate = xp.array(learning_rate, dtype=DTYPE)

        self.scheduler = None
        self.enable_weight_decay = False

    def get_config(self):
        from LunarLearn.engine import serialize_value

        init_params = ["learning_rate", "epsilon"]

        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "params": {
                k: serialize_value(v)
                for k, v in self.__dict__.items()
                if k in init_params
            },
            "extra": {
                k: serialize_value(v)
                for k, v in self.__dict__.items()
                if k not in init_params and k != "scheduler"
            },
            "scheduler": self.scheduler.get_config() if self.scheduler else None
        }

    @classmethod
    def from_config(cls, config):
        from LunarLearn.engine import object_from_config
        import importlib

        # Import the optimizer class
        module = importlib.import_module(config["module"])
        klass = getattr(module, config["class"])

        # Initialize with params
        obj = klass(**config.get("params", {}))

        # Set extra attributes
        for k, v in config.get("extra", {}).items():
            setattr(obj, k, v)

        # Restore scheduler if present
        if "scheduler" in config and config["scheduler"] is not None:
            obj.scheduler = object_from_config(config["scheduler"], optimizer=obj)

        return obj
    
    def _iter_params(self, params):
        """Yield trainable Parameter objects from a params list."""
        for desc in params:
            if isinstance(desc, dict):
                param = desc["param"]
                layer = desc.get("layer", None)
            else:
                param = desc
                layer = None

            if not isinstance(param, Parameter):
                continue
            if param.grad is None:
                continue

            # Skip frozen parameters
            if getattr(param, "frozen", False):
                continue
            if layer is not None and getattr(layer, "frozen", False):
                continue

            # Priority: per-param optimizer > per-layer optimizer > global optimizer
            custom_optim = param.optimizer or (layer.optimizer if layer else None)

            yield param, layer, custom_optim
    
    def _get_lr(self, param, layer=None):
        """
        Compute the effective learning rate for a parameter.

        Priority:
        1. param.base_lr > layer.base_lr > global LR
        2. param.lr_scale > layer.lr_scale > 1.0
        """
        # 1. Base LR priority chain
        if getattr(param, "base_lr", None) is not None:
            lr = param.base_lr
        elif layer is not None and getattr(layer, "base_lr", None) is not None:
            lr = layer.base_lr
        else:
            lr = self.learning_rate

        # 2. Scaling (multiplicative)
        lr *= getattr(param, "lr_scale", 1.0)
        if layer is not None:
            lr *= getattr(layer, "lr_scale", 1.0)

        return lr
    
    def _apply_weight_decay(self, param, layer=None, lr=None):
        """
        Applies decoupled weight decay to a parameter if configured.
        Priority: param.weight_decay > layer.weight_decay > optimizer.weight_decay
        """
        wd = 0.0

        if getattr(param, "decay_exempt", False):
            return
        if getattr(param, "frozen", False):
            return
        if layer is not None:
            if getattr(layer, "decay_exempt", False):
                return
            if getattr(layer, "frozen", False):
                return
        
        if not self.enable_weight_decay:
            return

        # 1. Per-param setting
        if getattr(param, "weight_decay", None) is not None:
            wd = param.weight_decay

        # 2. Per-layer setting
        elif layer is not None and getattr(layer, "weight_decay", None) is not None:
            wd = layer.weight_decay

        # 3. Global optimizer default
        elif hasattr(self, "weight_decay"):
            wd = self.weight_decay

        # If no weight decay, bail out
        if wd == 0.0:
            return
        
        wd *= getattr(param, "weight_decay_scale", 1.0)
        if layer is not None:
            wd *= getattr(layer, "weight_decay_scale", 1.0)

        # Resolve LR if not passed
        if lr is None:
            lr = self._get_lr(param, layer)

        # Decoupled weight decay update
        param.data -= lr * wd * param.data

    def step(self, params):
        """Update parameters based on their gradients."""
        raise NotImplementedError

    def zero_grad(self, params):
        """Set gradients to zero."""
        for p in params:
            if isinstance(p, Parameter) and p.requires_grad:
                p.grad = None