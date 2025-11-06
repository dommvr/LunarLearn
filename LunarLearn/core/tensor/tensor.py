import LunarLearn.core.backend.backend as backend
from ops import (
    # slicing
    slice, 

    # arithmetic ops
    add, subtract, multiply, divide, power, neg,

    # comparison ops
    eq, ne, lt, le, gt, ge,

    # logical ops
    logical_and, logical_or, logical_xor, logical_not,

    # linear algebra / shape ops
    matmul, reshape, transpose, expand,

    # reductions / stats
    mean, std, var, logsumexp,
    sum as sum_, max as max_, min as min_, abs as abs_
)
from utils import ensure_tensor

xp = backend.xp
DTYPE = backend.DTYPE

class Tensor:
    # ======================================================
    # Core initialization
    # ======================================================
    def __init__(self, data, requires_grad=False, dtype=None):
        """
        Tensor(data, requires_grad=False, dtype=None)

        Core tensor object for LunarLearn.

        Args:
            data: array-like, numpy.ndarray, cupy.ndarray, or scalar.
            requires_grad (bool): track gradients for autograd.
            dtype (str or np.dtype, optional): data type to cast input to.
        """
        import numpy as np

        if isinstance(data, Tensor):
            self.data = data.data.astype(dtype or data.dtype)
            self.requires_grad = requires_grad or data.requires_grad
        else:
            if xp.isscalar(data):
                data = xp.array(data, dtype=(dtype or DTYPE))
            elif isinstance(data, (np.ndarray, xp.ndarray)):
                data = xp.array(data, dtype=(dtype or DTYPE))
            else:
                data = xp.array(np.array(data), dtype=(dtype or DTYPE))

            self.data = data
            self.requires_grad = requires_grad and backend.is_grad_enabled()

        self.dtype = self.data.dtype
        self.grad = None
        self.grad_fn = None
        self._activation_hooks = []
        self._grad_hooks = []
        self._retain_grad = False
        self.is_leaf = True
        self._backward = lambda: None
        self._prev = set()

    # ======================================================
    # Display / Python integration
    # ======================================================
    def __repr__(self):
        """
        String representation with truncated array contents.
        Shows first few elements per dimension for readability.
        """
        def truncate(arr):
            if arr.ndim == 0:
                return str(arr.item())
            if arr.ndim == 1:
                s = arr[:3]
                return f"{s.tolist()}..." if arr.size > 3 else f"{s.tolist()}"
            s = arr[:3]
            rows = [truncate(row) for row in s]
            return "[" + ",\n ".join(rows) + ("..." if arr.shape[0] > 3 else "") + "]"

        data_str = truncate(self.data)
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad}, data={data_str})"

    def __len__(self):
        """Return length of first dimension. Raises TypeError for scalars."""
        if self.ndim == 0:
            raise TypeError("Scalar tensor has no length")
        return self.data.shape[0]

    def __bool__(self):
        """
        Convert to bool.
        Only valid for scalar tensors (size == 1).
        """
        if self.size != 1:
            raise ValueError("The truth value of a tensor with more than one element is ambiguous")
        return bool(self.data)

    def __iter__(self):
        """Iterate over the first dimension of the tensor."""
        if self.ndim == 0:
            raise TypeError("Iteration over a 0-d tensor")
        for i in range(len(self)):
            yield self[i]

    # ======================================================
    # Indexing
    # ======================================================
    def __getitem__(self, idx):
        """Tensor slicing. Delegates to ops.slice_op."""
        return slice(self, idx)

    # ======================================================
    # Arithmetic operators
    # ======================================================
    def __add__(self, other):
        """Elementwise addition (self + other)."""
        return add(self, ensure_tensor(other))
    def __radd__(self, other):
        """Elementwise addition (other + self)."""
        return add(ensure_tensor(other), self)

    def __sub__(self, other): return subtract(self, ensure_tensor(other))
    """Elementwise subtraction (self - other)."""
    def __rsub__(self, other): return subtract(ensure_tensor(other), self)
    """Elementwise subtraction (other - self)."""

    def __mul__(self, other): return multiply(self, ensure_tensor(other))
    """Elementwise multiplication (self * other)."""
    def __rmul__(self, other): return multiply(ensure_tensor(other), self)
    """Elementwise multiplication (other * self)."""

    def __truediv__(self, other): return divide(self, ensure_tensor(other))
    """Elementwise division (self / other)."""
    def __rtruediv__(self, other): return divide(ensure_tensor(other), self)
    """Elementwise division (other / self)."""

    def __pow__(self, other): return power(self, ensure_tensor(other))
    """Elementwise power (self ** other)."""
    def __rpow__(self, other): return power(ensure_tensor(other), self)
    """Elementwise power (other ** self)."""

    def __abs__(self): return abs_(self)
    """Absolute value (self)."""
    def __neg__(self): return neg(self)
    """Negative value (self)."""

    # ======================================================
    # Matrix multiplication
    # ======================================================
    def __matmul__(self, other):
        """Matrix multiplication using '@' operator."""
        return matmul(self, ensure_tensor(other))

    # ======================================================
    # Comparison operators
    # ======================================================
    def __eq__(self, other):
        """Elementwise equality (==)."""
        return eq(self, ensure_tensor(other))
    def __ne__(self, other):
        """Elementwise inequality (!=)."""
        return ne(self, ensure_tensor(other))
    def __lt__(self, other):
        """Elementwise less-than (<)."""
        return lt(self, ensure_tensor(other))
    def __le__(self, other):
        """Elementwise less-or-equal (<=)."""
        return le(self, ensure_tensor(other))
    def __gt__(self, other):
        """Elementwise greater-than (>)."""
        return gt(self, ensure_tensor(other))
    def __ge__(self, other):
        """Elementwise greater-or-equal (>=)."""
        return ge(self, ensure_tensor(other))

    # ======================================================
    # Logical operators
    # ======================================================
    def __and__(self, other):
        """Elementwise logical AND (&)."""
        return logical_and(self, ensure_tensor(other))
    def __rand__(self, other):
        """Elementwise logical AND (&)."""
        return logical_and(ensure_tensor(other), self)

    def __or__(self, other):
        """Elementwise logical OR (|)."""
        return logical_or(self, ensure_tensor(other))
    def __ror__(self, other):
        """Elementwise logical OR (|)."""
        return logical_or(ensure_tensor(other), self)

    def __xor__(self, other):
        """Elementwise logical XOR (^)."""
        return logical_xor(self, ensure_tensor(other))
    def __rxor__(self, other):
        """Elementwise logical XOR (^)."""
        return logical_xor(ensure_tensor(other), self)

    def __invert__(self):
        """Elementwise logical NOT (~)."""
        return logical_not(self)

    # ======================================================
    # Tensor methods: reshaping / views
    # ======================================================
    def reshape(self, *shape):
        """
        Returns a reshaped view of the tensor.
        Args:
            *shape: target shape (tuple or ints).
        """
        return reshape(self, shape if isinstance(shape[0], int) else shape[0])

    def flatten(self):
        """Flatten tensor into 1D view."""
        return self.reshape(self.size)

    @property
    def T(self):
        """Shorthand for transpose (last two dims swapped)."""
        return transpose(self)

    def expand(self, shape):
        """Broadcast tensor to new shape."""
        return expand(self, shape)

    # ======================================================
    # Reductions / stats
    # ======================================================
    def sum(self, axis=None, keepdims=False):
        """Sum of elements over axis."""
        return sum_(self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False):
        """Mean of elements over axis."""
        return mean(self, axis=axis, keepdims=keepdims)
    def max(self, axis=None, keepdims=False):
        """Maximum over axis."""
        return max_(self, axis=axis, keepdims=keepdims)
    def min(self, axis=None, keepdims=False):
        """Minimum over axis."""
        return min_(self, axis=axis, keepdims=keepdims)
    def std(self, axis=None, keepdims=False, eps=1e-8):
        """Standard deviation over axis."""
        return std(self, axis=axis, keepdims=keepdims, eps=eps)
    def var(self, axis=None, keepdims=False, eps=1e-8):
        """Variance over axis."""
        return var(self, axis=axis, keepdims=keepdims, eps=eps)
    def logsumexp(self, axis=None, keepdims=False):
        """Log-sum-exp trick for numerical stability."""
        return logsumexp(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        """Returns True if all elements evaluate to True."""
        return Tensor(xp.all(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
    def any(self, axis=None, keepdims=False):
        """Returns True if any element evaluates to True."""
        return Tensor(xp.any(self.data, axis=axis, keepdims=keepdims), requires_grad=False)

    # ======================================================
    # Conversion / utility
    # ======================================================
    def zero_grad(self):
        """Clear gradients (set to None)."""
        self.grad = None
    def detach(self):
        """Return a new Tensor detached from graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    def item(self):
        """Return Python scalar from a 0-dim Tensor."""
        if self.size != 1:
            raise ValueError("Can only convert scalar tensor to Python number")
        return self.data.item()
    def clone(self):
        """Return a copy of the tensor with same settings."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad, dtype=self.dtype)
    def numpy(self):
        """Return NumPy array (copy if GPU backend)."""
        return xp.asnumpy(self.data) if 'cupy' in xp.__name__ else self.data

    def float(self):
        """Cast to float32."""
        return self.astype('float32')
    def int(self):
        """Cast to int32."""
        return self.astype('int32')
    def astype(self, dtype, copy=True):
        """Return new Tensor with given dtype."""
        data = self.data.astype(dtype=dtype, copy=copy)
        out = Tensor(data, requires_grad=self.requires_grad, dtype=dtype)
        out.is_leaf = False
        out.grad_fn = "astype"
        out._prev = {self}
        return out
    
    def register_activation_hook(self, hook_fn):
        """
        Register a activation hook to be called when this tensor computes a data.

        Args:
            hook_fn (Callable): A function that takes a gradient Tensor and returns a (possibly modified) Tensor.
        Returns:
            The hook function (so it can be removed later if desired).
        """
        self._activation_hooks.append(hook_fn)
        return hook_fn

    def register_grad_hook(self, hook_fn):
        """
        Register a gradient hook to be called when this tensor receives a gradient.

        Args:
            hook_fn (Callable): A function that takes a gradient Tensor and returns a (possibly modified) gradient.
        Returns:
            The hook function (so it can be removed later if desired).
        """
        self._grad_hooks.append(hook_fn)
        return hook_fn

    # ======================================================
    # Properties
    # ======================================================
    @property
    def shape(self):
        """Tensor shape as tuple."""
        return self.data.shape
    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.data.shape)
    @property
    def size(self):
        """Number of elements."""
        return self.data.size

    # ======================================================
    # Constructors
    # ======================================================
    @classmethod
    def randn(cls, shape, requires_grad=False, dtype=None):
        """Return Tensor with values from N(0,1)."""
        return cls(xp.random.randn(*shape).astype(dtype or DTYPE), requires_grad, dtype)
    @classmethod
    def zeros(cls, shape, requires_grad=False, dtype=None):
        """Return Tensor filled with zeros."""
        return cls(xp.zeros(shape, dtype=(dtype or DTYPE)), requires_grad, dtype)
    @classmethod
    def ones(cls, shape, requires_grad=False, dtype=None):
        """Return Tensor filled with ones."""
        return cls(xp.ones(shape, dtype=(dtype or DTYPE)), requires_grad, dtype)
    @classmethod
    def full(cls, shape, fill_value, requires_grad=False, dtype=None):
        """Return Tensor filled with a scalar value."""
        return cls(xp.full(shape, fill_value, dtype=(dtype or DTYPE)), requires_grad, dtype)
    @classmethod
    def eye(cls, n, requires_grad=False, dtype=None):
        """Return identity matrix of size (n,n)."""
        return cls(xp.eye(n, dtype=(dtype or DTYPE)), requires_grad, dtype)

    # ======================================================
    # Device / dtype movement
    # ======================================================
    def to(self, device=None, dtype=None):
        """
        Move tensor to device (cpu/cuda) and/or cast dtype.
        Args:
            device (str): 'cpu' or 'cuda'
            dtype (np.dtype or str, optional)
        """
        data = self.data
        if dtype is not None:
            data = data.astype(dtype)
        if device is not None:
            if device == "cpu" and "cupy" in xp.__name__:
                import cupy; data = cupy.asnumpy(data)
            elif device == "cuda" and "numpy" in xp.__name__:
                import cupy; data = cupy.array(data)
            # if already on cuda with cupy, nothing changes
        return Tensor(data, requires_grad=self.requires_grad, dtype=dtype or self.dtype)

    def cpu(self, dtype=None):
        """Move tensor to CPU."""
        return self.to(device='cpu', dtype=dtype)
    def cuda(self, dtype=None):
        """Move tensor to CUDA (if available)."""
        return self.to(device='cuda', dtype=dtype)

    # ======================================================
    # Autograd
    # ======================================================
    def requires_grad_(self, requires=True):
        """
        Set requires_grad in-place.
        Args:
            requires (bool): track gradients if True.
        """
        self.requires_grad = requires
        return self

    def retain_grad(self):
        """Retain grad for non-leaf tensors."""
        if not self.requires_grad:
            raise RuntimeError("Cannot retain grad on a tensor that does not require grad")
        self._retain_grad = True
        return self

    def visualize(self, mode="text", indent=0, visited=None):
        """
        Visualize the autograd graph.
        mode="text" -> prints a tree structure
        mode="graph" -> renders with Graphviz (requires graphviz installed)
        """
        if mode == "text":
            if visited is None:
                visited = set()

            prefix = "  " * indent
            grad_info = f"requires_grad={self.requires_grad}"
            node_type = f"Tensor(shape={self.shape}, dtype={self.dtype})"

            if self.grad_fn:
                node_type += f" <- {self.grad_fn}"

            print(f"{prefix}{node_type} [{grad_info}]")

            if self in visited:
                print(f"{prefix}  ↳ (already visited)")
                return
            visited.add(self)

            for p in self._prev:
                p.visualize(mode="text", indent=indent + 1, visited=visited)

        elif mode == "graph":
            try:
                import graphviz

                dot = graphviz.Digraph()

                def add_nodes(t):
                    if t not in dot.node_attr:
                        label = f"{t.grad_fn or 'Leaf'}\n{t.shape}, {t.dtype}"
                        dot.node(str(id(t)), label)
                        for p in t._prev:
                            dot.edge(str(id(p)), str(id(t)))
                            add_nodes(p)

                add_nodes(self)
                return dot
            except ImportError:
                raise ImportError("Graphviz is not installed. Run `pip install graphviz`")
        else:
            raise ValueError("mode must be 'text' or 'graph'")

    def backward(self, grad=None):
        """
        Backpropagate gradients through computation graph.

        Args:
            grad: initial gradient (defaults to ones for scalar).
        """
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("Grad must be specified for non-scalar outputs")
            grad = xp.ones_like(self.data, dtype=self.dtype)
        self.grad = grad.astype(self.dtype, copy=False)

        topo, visited = [], set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    if child.requires_grad:
                        build_topo(child)
                topo.append(t)

        if not self.requires_grad:
            return

        build_topo(self)

        for t in reversed(topo):
            if not t.requires_grad:
                continue
            t._backward()
            if not (t.is_leaf or t._retain_grad):
                t.grad = None