from .tensor import Tensor
import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Stateful

xp = backend.xp

class Parameter(Stateful):
    """
    Trainable parameter with automatic AMP support:
    - Stores a master FP32 weight
    - Maintains a compute FP16 copy for forward ops
    - Casts grads back to FP32 after backward
    """
    def __init__(self, data, requires_grad=True):
        # Always store master weight in FP32
        self.master = Tensor(data.astype(backend.DTYPE, copy=False), requires_grad=requires_grad)
        self.compute = self.master.astype(backend.C_DTYPE, copy=False)
        self.requires_grad = requires_grad

        self.normalization = None
        self.regularizer = None
        self.weight_decay = None
        self.weight_decay_scale = 1.0
        self.decay_exempt = False

        self.lr_scale = 1.0
        self.base_lr = None
        self.optimizer = None
        self.scheduler = None

        self.frozen = False

        self._state_fields = [
            "requires_grad",
            "weight_decay",
            "weight_decay_scale",
            "decay_exempt",
            "lr_scale",
            "base_lr",
            "frozen"
        ]

    def state_dict(self):
        out = {
            "master_data": self.master.data,
            "master_grad": self.master.grad,
            "dtype": str(self.master.dtype) if self.master.dtype is not None else None,
        }

         # optional attached states
        if self.normalization is not None:
            out["normalization"] = self.normalization.state_dict()
        if self.regularizer is not None:
            out["regularizer"] = self.regularizer.state_dict()
        if self.optimizer is not None:
            out["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            out["scheduler"] = self.scheduler.state_dict()

        for name in self._state_fields:
            val = getattr(self, name, None)
            if val is not None:
                out[name] = val
        
        return out
    
    def load_state_dict(self, state):
        # restore tensor data safely
        if "master_data" in state:
            self.master.data[...] = xp.array(state["master_data"])

        # restore grad if present
        grad = state.get("master_grad", None)
        if grad is not None:
            self.master.grad = xp.array(grad)

        # load submodules if present and initialized
        if "normalization" in state and self.normalization is not None:
            self.normalization.load_state_dict(state["normalization"])
        if "regularizer" in state and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])

        if "optimizer" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])

        # restore scalars
        for name in self._state_fields:
            if name in state:
                setattr(self, name, state[name])
        
    @property
    def data(self):
        """Access the master weight data."""
        return self.master.data
    
    @property
    def grad(self):
        return self.master.grad
    
    @property
    def shape(self):
        return self.master.shape
    
    def register_activation_hook(self, hook_fn):
        self.master.register_activation_hook(hook_fn)
        if getattr(self, "compute", None) is not None:
            self.compute.register_activation_hook(hook_fn)
        return hook_fn

    def register_grad_hook(self, hook_fn):
        self.master.register_grad_hook(hook_fn)
        if getattr(self, "compute", None) is not None:
            self.compute.register_grad_hook(hook_fn)
        return hook_fn
    
    def parameters(self):
        return [p for _, p in self.named_parameters()]
    
    def named_parameters(self, prefix: str = ""):
        params = []
        # Add this parameter itself
        params.append((prefix, self))

        # If normalization exists, recurse into it
        norm = getattr(self, "normalization", None)
        if norm is not None and hasattr(norm, "named_parameters"):
            for n, p in norm.named_parameters(prefix=f"{prefix}.norm"):
                params.append((n, p))
        return params

    def to_compute(self):
        """Return a compute copy for forward pass (FP16 if AMP enabled)."""
        if backend.MIXED_PRECISION:
            self.compute = self.master.astype(backend.C_DTYPE, copy=False)
        else:
            self.compute = self.master.clone()

        if self.normalization is not None:
            self.compute = self.normalization(self.compute)

        self.compute.register_grad_hook(self._sync_grad)
        return self.compute

    def _sync_grad(self, grad):
        if grad is not None:
            self.master.grad = grad.astype(backend.DTYPE, copy=False)
        return grad  # must return grad for normal flow

    def zero_grad(self):
        """Clear master gradient."""
        self.master.grad = None

    def __repr__(self):
        return f"Parameter(master={self.master}, compute={self.compute}, grad={self.master.grad})"
