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

        self.regularizer = None
        self.weight_decay = None
        self.weight_decay_scale = 1.0
        self.decay_exempt = False

        self.lr_scale = 1.0
        self.base_lr = None
        self.optimizer = None
        self.scheduler = None

        self.frozen = False
        
        self.compute.register_hook(self._sync_grad)

    @property
    def data(self):
        """Access the master weight data."""
        return self.master.data
    
    @property
    def grad(self):
        return self.master.grad

    def to_compute(self):
        """Return a compute copy for forward pass (FP16 if AMP enabled)."""
        if backend.MIXED_PRECISION:
            self.compute = self.master.astype(backend.C_DTYPE, copy=False)
        else:
            self.compute = self.master.clone()
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
