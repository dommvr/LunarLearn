import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Stateful
from LunarLearn.core import Tensor, Parameter

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE

class BaseLayer(Stateful):

    def __init__(self, trainable: bool = False):
        self.trainable = trainable
        self.training = True
        
        # Parameters
        self.W, self.b = None, None

        # Forward-pass cache
        self.A, self.Z, self.D = None, None, None

        # Shape/hyperparameters
        self.nodes = None
        self.shape, self.input_shape, self.output_shape = None, None, None
        self.kernel_size, self.filters = None, None
        self.strides, self.padding = None, None
        self.n_C, self.n_H, self.n_W = None, None, None

        # Config
        self.activation, self.w_init = None, None
        self.uniform, self.gain, self.keep_prob = None, None, None

        # LR tricks
        self.lr_scale = 1.0
        self.base_lr = None
        self.optimizer = None
        self.scheduler = None

        self.frozen = False

        self.param_settings = {}

        # Hook system
        self.hooks = {}
        self.custom_hook_metrics = []

        self.regularizer = None
        self.weight_decay = None
        self.weight_decay_scale = 1.0
        self.decay_exempt = False

        self._state_fields = [
            "training",
            "lr_scale",
            "base_lr",
            "frozen",
            "weight_decay",
            "weight_decay_scale",
            "decay_exempt"
        ]

    def is_initialized(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Parameter):
                return True
        return False

    def _named_state_items(self):
        for name, v in self.__dict__.items():
            if isinstance(v, Stateful):
                yield name, v

    def state_dict(self):
        out = {"_type": self.__class__.__name__}
        for name, obj in self._named_state_items():
            out[name] = obj.state_dict()
        for name in self._state_fields:
            val = getattr(self, name, None)
            if val is not None:
                out[name] = val
        return out
    
    def load_state_dict(self, state):
        if not self.is_initialized():
            self._pending_state = state
            return
        
        for name, obj in self._named_state_items():
            if name in state:
                obj.load_state_dict(state[name])

        for name in self._state_fields:
            if name in state:
                setattr(self, name, state[name])

    def apply_pending_state_if_any(self):
        ps = getattr(self, "_pending_state", None)
        if ps is None:
            return
        del self._pending_state
        self.load_state_dict(ps)

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(x, *args, **kwargs)
    
    def __repr__(self):
        class_name = self.__class__.__name__
        
        # If this layer contains submodules (e.g. Sequential/Model), recurse
        if hasattr(self, "layers") and isinstance(self.layers, (list, tuple)):
            child_str = "\n".join(
                ["  (" + str(i) + "): " + repr(layer).replace("\n", "\n    ")
                for i, layer in enumerate(self.layers)]
            )
            return f"{class_name}(\n{child_str}\n)"
        else:
            # Layer-specific extra info
            extra = self.extra_repr()
            if extra:
                return f"{class_name}({extra})"
            else:
                shape_str = f"in={self.input_shape}, out={self.output_shape}"
                params = sum(p.data.size for p in self.parameters())
                return f"{class_name}({shape_str}, params={params})"
            
    def extra_repr(self) -> str:
        """
        Override in subclasses to provide custom layer-specific
        information for __repr__. By default, shows input/output shape.
        """
        if self.input_shape and self.output_shape:
            return f"in={self.input_shape}, out={self.output_shape}"
        return ""
    
    def count_parameters(self) -> int:
        """Return the total number of trainable parameters in this layer."""
        total = 0
        for name in ("W", "b"):
            param = getattr(self, name, None)
            if param is not None and hasattr(param, "data"):
                total += param.data.size
        return total

    def set_param(self, name, **kwargs):
        """Register settings for a parameter that might not exist yet."""
        self.param_settings.setdefault(name, {}).update(kwargs)
        return self  # allow chaining

    def _apply_param_settings(self):
        for name, settings in self.param_settings.items():
            param = getattr(self, name, None)
            if param is None:
                continue
            for k, v in settings.items():
                if k == "normalization" and v is not None:
                    if isinstance(v, tuple):
                        cls, kwargs = v
                        v = cls(param, **kwargs)
                    elif isinstance(v, type):
                        v = v(param)
                    setattr(param, k, v)
                else:
                    setattr(param, k, v)
    
    def train(self):
        """
        Set this layer to training mode.

        Layers like Dropout or BatchNorm will use this flag to
        change their forward-pass behavior. By default, just sets
        the internal `training` flag to True.
        """
        self.training = True

    def eval(self):
        """
        Set this layer to evaluation mode.

        Layers like Dropout or BatchNorm will use this flag to
        change their forward-pass behavior. By default, just sets
        the internal `training` flag to False.
        """
        self.training = False

    def freeze(self):
        """Freeze all parameters in this layer."""
        for p in self.parameters():
            p.frozen = True

    def unfreeze(self):
        """Unfreeze all parameters in this layer."""
        for p in self.parameters():
            p.frozen = False

    # -------------------------------
    # Parameter collection
    # -------------------------------
    def parameters(self, with_layer: bool = False): #layer
        return [p for _, p in self.named_parameters(with_layer=with_layer)]

    def named_parameters(self, prefix: str = "", with_layer: bool = False): #layer
        params = []
        for name, v in self.__dict__.items():
            if isinstance(v, Parameter):
                pname = f"{prefix}{name}"
                for n, p in v.named_parameters(pname):
                    params.append((n, {"param": p, "layer": self} if with_layer else v))
                        
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if isinstance(item, Parameter):
                        pname = f"{prefix}{name}{i}"
                        for n, p in item.named_parameters(pname):
                            params.append((n, {"param": p, "layer": self} if with_layer else item))
        return params

    # -------------------------------
    # Abstracts (implemented in child)
    # -------------------------------
    def initialize(self, input_shape):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError