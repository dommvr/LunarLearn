from LunarLearn.tensor import Tensor
import LunarLearn.backend as backend

class Parameter:
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
        
    @property
    def data(self):
        """Access the master weight data."""
        return self.master.data
    
    @property
    def grad(self):
        return self.master.grad
    
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
