import LunarLearn.core.backend.backend as backend
from .tensor import Tensor
from .utils import ensure_tensor, unbroadcast, promote_dtype
from LunarLearn.amp import dispatch_amp

xp = backend.xp
DTYPE = backend.DTYPE

# ============================================================================
# Creation operations
# ============================================================================

def _zeros_impl(shape, dtype=DTYPE) -> Tensor:
    data = xp.zeros(shape, dtype=dtype)
    out = Tensor(data, requires_grad=False, dtype=dtype)
    out.is_leaf = True
    out.grad_fn = "zeros"
    return out

def zeros(shape, dtype=DTYPE) -> Tensor:
    """
    Create a Tensor filled with zeros.

    Args:
        shape (tuple): Desired tensor shape.
        dtype (dtype): Data type of the tensor.

    Returns:
        Tensor: New tensor of given shape filled with zeros.
    """
    return dispatch_amp("zeros", _zeros_impl, shape, dtype=dtype)

def _ones_impl(shape, dtype=DTYPE) -> Tensor:
    data = xp.ones(shape, dtype=dtype)
    out = Tensor(data, requires_grad=False, dtype=dtype)
    out.is_leaf = True
    out.grad_fn = "ones"
    return out

def ones(shape, dtype=DTYPE) -> Tensor:
    """
    Create a Tensor filled with ones.

    Args:
        shape (tuple): Desired tensor shape.
        dtype (dtype): Data type of the tensor.

    Returns:
        Tensor: New tensor of given shape filled with ones.
    """
    return dispatch_amp("ones", _ones_impl, shape, dtype=dtype)

def _full_impl(shape, value, dtype=DTYPE) -> Tensor:
    data = xp.full(shape, value, dtype=DTYPE)
    out = Tensor(data, requires_grad=False, dtype=dtype)
    out.is_leaf = True
    out.grad_fn = 'full'
    return out

def full(shape, value, dtype=DTYPE) -> Tensor:
    """
    Create a Tensor filled with a specified value.

    Args:
        shape (tuple): Desired tensor shape.
        value (scalar): Constant value to fill the tensor with.
        dtype (dtype, optional): Data type of the tensor. Default is DTYPE.

    Returns:
        Tensor: New tensor of given shape filled with `value`.
    """
    return dispatch_amp("full", _full_impl, shape, value, dtype=dtype)

def _arange_impl(*args, dtype=DTYPE) -> Tensor:
    data = xp.arange(*args, dtype=dtype)
    out = Tensor(data, requires_grad=False, dtype=dtype)
    out.is_leaf = True
    out.grad_fn = "arange"
    return out

def arange(*args, dtype=DTYPE) -> Tensor:
    """
    Create a 1D tensor with values from a range.

    Args:
        *args: Same semantics as Python's range or numpy.arange.
        dtype (dtype): Data type of the tensor.

    Returns:
        Tensor: 1D tensor with evenly spaced values.
    """
    return dispatch_amp("arange", _arange_impl, *args, dtype=dtype)

def _eye_impl(n, m=None, dtype=DTYPE) -> Tensor:
    data = xp.eye(N=n, M=m if m is not None else n, dtype=dtype)
    out = Tensor(data, requires_grad=False, dtype=dtype)
    out.is_leaf = True
    out.grad_fn = "eye"
    return out

def eye(n, m=None, dtype=DTYPE):
    """
    Create a 2D identity matrix (or rectangular eye matrix).

    Args:
        n (int): Number of rows.
        m (int, optional): Number of columns. If None, defaults to n.
        dtype (dtype): Data type of the tensor.

    Returns:
        Tensor: Identity (or eye) tensor.
    """
    return dispatch_amp("eye", _eye_impl, n, m, dtype=dtype)

# ============================================================================
# Unary operations
# ============================================================================

def unary_op(a: Tensor, op, grad_fn, name):
    """
    General helper for unary operations with autograd support.

    Args:
        a (Tensor): Input tensor.
        op (callable): Forward operation applied to `a.data`.
        grad_fn (callable): Function computing gradient wrt input.
        name (str): Name of the operation.

    Returns:
        Tensor: Result tensor with autograd tracking.
    """
    a = ensure_tensor(a)
    dtype = a.dtype
    data = op(a.data.astype(dtype))
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, 
                 requires_grad=requires_grad,
                 dtype=dtype)
    out.is_leaf = False
    out.grad_fn = name

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out
    
    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad.astype(dtype, copy=False)
        grad_a = grad_fn(grad_out, a.data)[0]
        grad_a = unbroadcast(grad_a, a.shape)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def _exp_impl(a: Tensor) -> Tensor:
    return unary_op(
        a,
        op=xp.exp,
        grad_fn=lambda grad_out, a_data: grad_out * xp.exp(a_data),
        name="exp",
    )

def exp(a: Tensor) -> Tensor:
    """
    Elementwise exponential.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: exp(a).
    """
    return dispatch_amp("exp", _exp_impl, a)

def _log_impl(a: Tensor) -> Tensor:
    return unary_op(
        a,
        op=xp.log,
        grad_fn=lambda grad_out, a_data: grad_out / a_data,
        name="log",
    )

def log(a: Tensor) -> Tensor:
    """
    Elementwise natural logarithm.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: log(a).
    """
    return dispatch_amp("log", _log_impl, a)

def _sqrt_impl(a: Tensor) -> Tensor:
    return unary_op(
        a,
        op=xp.sqrt,
        grad_fn=lambda grad_out, a_data: grad_out / (2 * xp.sqrt(a_data)),
        name="sqrt",
    )

def sqrt(a: Tensor) -> Tensor:
    """
    Elementwise square root.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: sqrt(a).
    """
    return dispatch_amp("sqrt", _sqrt_impl, a)

def _abs_impl(a: Tensor) -> Tensor:
    return unary_op(
        a,
        op=xp.abs,
        grad_fn=lambda grad_out, a_data: (grad_out * xp.sign(a_data),),
        name="abs"
    )

def abs(a: Tensor) -> Tensor:
    """
    Elementwise absolute value.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: Absolute value of `a`.
    """
    return dispatch_amp("abs", _abs_impl, a)

def _neg_impl(a: Tensor) -> Tensor:
    return unary_op(
        a,
        op=lambda x: -x,
        grad_fn=lambda grad_out, a_data: (-grad_out,),
        name="neg"
    )

def neg(a: Tensor) -> Tensor:
    """
    Elementwise negation.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: Negated tensor (-a).
    """
    return dispatch_amp("neg", _neg_impl, a)

def _norm_impl(a: Tensor, ord=2, axis=None, keepdims=False, eps=1e-8) -> Tensor:
    a = ensure_tensor(a)
    squared = (a * a).sum(axis=axis, keepdims=keepdims)
    data = xp.sqrt(squared + eps)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "norm"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_a = out.grad * a.data / (data + eps)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def norm(a: Tensor, ord=2, axis=None, keepdims=False, eps=1e-8) -> Tensor:
    """
    Compute the L2 norm of a tensor along specified axes.

    Args:
        a (Tensor): Input tensor.
        ord (int or float, optional): Order of the norm. Default is 2.
        axis (int or tuple, optional): Axis or axes along which to compute the norm.
        keepdims (bool, optional): Whether to keep reduced dimensions. Default is False.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-8.

    Returns:
        Tensor: Tensor containing the computed norm.
    """
    return dispatch_amp("norm", _norm_impl, a, ord=ord, axis=axis, keepdims=keepdims, eps=eps)

def _l2_normalize_impl(a: Tensor, axis: int = -1, eps: float = 1e-8) -> Tensor:
    a = ensure_tensor(a)
    norm_val = norm(a, axis=axis, keepdims=True, eps=eps)
    return a / norm_val

def l2_normalize(a: Tensor, axis: int = -1, eps: float = 1e-8) -> Tensor:
    """
    L2-normalize tensor along specified axis.
    Equivalent to: a / (||a||_2 + eps)
    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axis or axes along which to compute the norm.
        keepdims (bool, optional): Whether to keep reduced dimensions. Default is False.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-8.

    Returns:
        Tensor: Tensor normalized along specified axis.
    """
    return dispatch_amp("l2_normalize", _l2_normalize_impl, a, axis=axis, eps=eps)

def _clip_impl(a: Tensor, a_min, a_max) -> Tensor:
    a = ensure_tensor(a)
    data = xp.clip(a.data, a_min, a_max)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "clip"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_a = out.grad * ((a.data >= a_min) & (a.data <= a_max))
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def clip(a: Tensor, a_min, a_max) -> Tensor:
    """
    Clip values of a tensor to a specified range.

    Args:
        a (Tensor): Input tensor.
        a_min (scalar): Minimum allowed value.
        a_max (scalar): Maximum allowed value.

    Returns:
        Tensor: Tensor with values clipped between a_min and a_max.
    """
    return dispatch_amp("clip", _clip_impl, a, a_min=a_min, a_max=a_max)

def _erf_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    data = xp.erf(a.data)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "erf"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        # derivative: d/dx erf(x) = 2/sqrt(pi) * exp(-x^2)
        grad_a = out.grad * (2.0 / xp.sqrt(xp.pi)) * xp.exp(-(a.data ** 2))
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def erf(a: Tensor) -> Tensor:
    """
    Elementwise Gauss error function.

    Computes:
        erf(a) = 2/sqrt(pi) * ∫_0^a exp(-t^2) dt

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: erf(a).
    """
    return dispatch_amp("erf", _erf_impl, a)

def _erfc_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    data = xp.erfc(a.data)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "erfc"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        # derivative of erfc(x) = -2/sqrt(pi) * exp(-x^2)
        grad_a = out.grad * (-2.0 / xp.sqrt(xp.pi)) * xp.exp(-(a.data ** 2))
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def erfc(a: Tensor) -> Tensor:
    """
    Elementwise complementary error function.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: erfc(a) = 1 - erf(a).
    """
    return dispatch_amp("erfc", _erfc_impl, a)

# ============================================================
# Binary elementwise operations
# ============================================================

def elementwise_op(a: Tensor, b: Tensor, op, grad_fn, name):
    """
    General helper for elementwise operations with autograd support.

    Args:
        a (Tensor): First tensor.
        b (Tensor): Second tensor.
        op (callable): Forward operation applied to `a.data` and `b.data`.
        grad_fn (callable): Function computing gradient wrt input.
        name (str): Name of the operation.

    Returns:
        Tensor: Result tensor with autograd tracking.
    """
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    dtype = promote_dtype(a.dtype, b.dtype)

    data = op(a.data.astype(dtype), b.data.astype(dtype))
    requires_grad = a.requires_grad or b.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = name

    for t in (a, b):
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        if not (a.requires_grad or b.requires_grad):
            return
        
        grad_out = out.grad.astype(dtype, copy=False)

        if a.requires_grad:
            grad_a, _ = grad_fn(grad_out, a.data, b.data)
            grad_a = unbroadcast(grad_a, a.shape)
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad += grad_a

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

        if b.requires_grad:
            _, grad_b = grad_fn(grad_out, a.data, b.data)
            grad_b = unbroadcast(grad_b, b.shape)
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad += grad_b

            for hook in getattr(b, "_grad_hooks", []):
                new_grad = hook(b.grad)
                if new_grad is not None:
                    b.grad = new_grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def _add_impl(a: Tensor, b: Tensor) -> Tensor:
    return elementwise_op(
        a, b,
        op=xp.add,
        grad_fn=lambda grad_out, a_data, b_data: (grad_out, 
                                                  grad_out),
                                                  name="add"
    )

def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise addition: out = a + b

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Result of a + b, with autograd support.
    """
    return dispatch_amp("add", _add_impl, a, b)

def _subtract_impl(a: Tensor, b: Tensor) -> Tensor:
    return elementwise_op(
        a, b,
        op=xp.subtract,
        grad_fn=lambda grad_out, a_data, b_data: (grad_out, 
                                                  -grad_out),
                                                  name="subtract"
    )

def subtract(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise subtraction: out = a - b

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Result of a - b, with autograd support.
    """
    return dispatch_amp("subtract", _subtract_impl, a, b)

def _multiply_impl(a: Tensor, b: Tensor) -> Tensor:
    return elementwise_op(
        a, b,
        op=xp.multiply,
        grad_fn=lambda grad_out, a_data, b_data: (grad_out * b_data, 
                                                  grad_out * a_data),
                                                  name="multiply"
    )

def multiply(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise multiplication: out = a * b

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Result of a * b, with autograd support.
    """
    return dispatch_amp("multiply", _multiply_impl, a, b)

def _divide_impl(a: Tensor, b: Tensor) -> Tensor:
    return elementwise_op(
        a, b,
        op=xp.divide,
        grad_fn=lambda grad_out, a_data, b_data: (grad_out / b_data, 
                                                  -grad_out * a_data / (b_data**2)),
                                                  name="divide"
    )

def divide(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise division: out = a / b

    Args:
        a (Tensor): Numerator.
        b (Tensor): Denominator.

    Returns:
        Tensor: Result of a / b, with autograd support.
    """
    return dispatch_amp("divide", _divide_impl, a, b)

def _power_impl(a: Tensor, b: Tensor) -> Tensor:
    return elementwise_op(
        a, b,
        op=xp.power,
        grad_fn=lambda grad_out, a_data, b_data: (grad_out * b_data * a_data ** (b_data-1), 
                                                  grad_out * (a_data ** b_data) * xp.log(a_data)),
                                                  name="power"
    )

def power(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise power: out = a ** b

    Args:
        a (Tensor): Base.
        b (Tensor): Exponent.

    Returns:
        Tensor: Result of a ** b, with autograd support.
    """
    return dispatch_amp("power", _power_impl, a, b)

def _maximum_impl(a: Tensor, b: Tensor) -> Tensor:
    return elementwise_op(
        a, b,
        op=xp.maximum,
        grad_fn=lambda grad_out, a_data, b_data: (
            grad_out * (a_data >= b_data),
            grad_out * (b_data > a_data),
        ),
        name="maximum"
    )

def maximum(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise maximum between two tensors.

    Args:
        a (Tensor): First input tensor.
        b (Tensor): Second input tensor.

    Returns:
        Tensor: Elementwise maximum of `a` and `b`.

    Notes:
        - Gradient w.r.t. `a` is passed only where a >= b.
        - Gradient w.r.t. `b` is passed only where b > a.
    """
    return dispatch_amp("maximum", _maximum_impl, a, b)

def _minimum_impl(a: Tensor, b: Tensor) -> Tensor:
    return elementwise_op(
        a, b,
        op=xp.minimum,
        grad_fn=lambda grad_out, a_data, b_data: (
            grad_out * (a_data <= b_data),
            grad_out * (b_data < a_data),
        ),
        name="minimum"
    )

def minimum(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise minimum between two tensors.

    Args:
        a (Tensor): First input tensor.
        b (Tensor): Second input tensor.

    Returns:
        Tensor: Elementwise minimum of `a` and `b`.

    Notes:
        - Gradient w.r.t. `a` is passed only where a <= b.
        - Gradient w.r.t. `b` is passed only where b < a.
    """
    return dispatch_amp("minimum", _minimum_impl, a, b)

def _where_impl(condition: Tensor, a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    dtype = promote_dtype(a.dtype, b.dtype)
    data = xp.where(condition.data, a.data, b.data)
    requires_grad = a.requires_grad or b.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "where"

    for t in (a, b):
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        if not (a.requires_grad or b.requires_grad):
            return
        
        grad_out = out.grad

        if a.requires_grad:
            grad_a = grad_out * condition.data
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad += grad_a

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

        if b.requires_grad:
            grad_b = grad_out * (~condition.data)
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad += grad_b

            for hook in getattr(b, "_grad_hooks", []):
                new_grad = hook(b.grad)
                if new_grad is not None:
                    b.grad = new_grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def where(condition: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """
    Select elements from `a` or `b` depending on a boolean condition.

    Args:
        condition (Tensor): Boolean tensor where True selects from `a` and False from `b`.
        a (Tensor): Tensor to select from when condition is True.
        b (Tensor): Tensor to select from when condition is False.

    Returns:
        Tensor: Tensor formed by choosing elements from `a` or `b`.
    """
    return dispatch_amp("where", _where_impl, condition, a, b)

# ============================================================
# Comparison operations
# ============================================================

def _eq_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(a.data == b.data, requires_grad=False)

def eq(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise equality: out = (a == b)

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Boolean mask (cast to int for autograd graph).
    """
    return dispatch_amp("eq", _eq_impl, a, b)

def _ne_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(a.data != b.data, requires_grad=False)

def ne(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise inequality: out = (a != b)

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Boolean mask (cast to int for autograd graph).
    """
    return dispatch_amp("ne", _ne_impl, a, b)

def _lt_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(a.data < b.data, requires_grad=False)

def lt(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise less-than: out = (a < b)

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Boolean mask (cast to int for autograd graph).
    """
    return dispatch_amp("lt", _lt_impl, a, b)

def _le_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(a.data <= b.data, requires_grad=False)

def le(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise less-equal: out = (a <= b)

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Boolean mask (cast to int for autograd graph).
    """
    return dispatch_amp("le", _le_impl, a, b)

def _gt_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(a.data > b.data, requires_grad=False)

def gt(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise greater-than: out = (a > b)

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Boolean mask (cast to int for autograd graph).
    """
    return dispatch_amp("gt", _gt_impl, a, b)

def _ge_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(a.data >= b.data, requires_grad=False)

def ge(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise greater-equal: out = (a >= b)

    Args:
        a (Tensor): First operand.
        b (Tensor): Second operand.

    Returns:
        Tensor: Boolean mask (cast to int for autograd graph).
    """
    return dispatch_amp("ge", _ge_impl, a, b)

# ============================================================
# Logical operations
# ============================================================

def _logical_and_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(xp.logical_and(a.data, b.data), requires_grad=False)

def logical_and(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise logical AND.

    Args:
        a (Tensor): First boolean tensor.
        b (Tensor): Second boolean tensor.

    Returns:
        Tensor: Boolean tensor with result of `a AND b`.

    Notes:
        - No gradients are propagated (non-differentiable).
    """
    return dispatch_amp("logical_and", _logical_and_impl, a, b)

def _logical_or_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(xp.logical_or(a.data, b.data), requires_grad=False)

def logical_or(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise logical OR.

    Args:
        a (Tensor): First boolean tensor.
        b (Tensor): Second boolean tensor.

    Returns:
        Tensor: Boolean tensor with result of `a OR b`.

    Notes:
        - No gradients are propagated (non-differentiable).
    """
    return dispatch_amp("logical_or", _logical_or_impl, a, b)

def _logical_xor_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return Tensor(xp.logical_xor(a.data, b.data), requires_grad=False)

def logical_xor(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise logical XOR (exclusive OR).

    Args:
        a (Tensor): First boolean tensor.
        b (Tensor): Second boolean tensor.

    Returns:
        Tensor: Boolean tensor with result of `a XOR b`.

    Notes:
        - No gradients are propagated (non-differentiable).
    """
    return dispatch_amp("logical_xor", _logical_xor_impl, a, b)

def _logical_not_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    return Tensor(xp.logical_not(a.data), requires_grad=False)

def logical_not(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise logical NOT.

    Args:
        a (Tensor): Input boolean tensor.

    Returns:
        Tensor: Boolean tensor with result of `NOT a`.

    Notes:
        - No gradients are propagated (non-differentiable).
    """
    return dispatch_amp("logical_not", _logical_not_impl, a, b)

# ============================================================
# Reduction operations
# ============================================================

def _sum_impl(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = ensure_tensor(a)
    data = xp.sum(a.data, axis=axis, keepdims=keepdims)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "sum"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        if axis is not None and not keepdims:
            grad_out = xp.expand_dims(grad_out, axis=axis)
        if a.requires_grad:
            grad_a = xp.broadcast_to(grad_out, a.shape)
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad += grad_a

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def sum(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """
    Reduction sum: out = sum(a)

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axis/axes to reduce.
        keepdims (bool): Whether to retain reduced dims.

    Returns:
        Tensor: Summed tensor.
    """
    return dispatch_amp("sum", _sum_impl, a, axis=axis, keepdims=keepdims)

def _cumsum_impl(a: Tensor, axis: int = -1) -> Tensor:
    a = ensure_tensor(a)
    data = xp.cumsum(a.data, axis=axis)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "cumsum"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad = out.grad
        if axis != -1:
            grad = xp.moveaxis(grad, axis, -1)
        rev_cumsum = xp.cumsum(grad[..., ::-1], axis=-1)[..., ::-1]
        if axis != -1:
            rev_cumsum = xp.moveaxis(rev_cumsum, -1, axis)
        if a.grad is None:
            a.grad = rev_cumsum
        else:
            a.grad += rev_cumsum

    out._backward = _backward
    out._prev = {a}
    return out

def cumsum(a: Tensor, axis: int = -1) -> Tensor:
    """
    Cumulative sum along specified dimension.

    Args:
        x (Tensor): Input tensor.
        axis (int, optional): Axis to compute cumulative sum. Default: -1.

    Returns:
        Tensor: Cumulative sum tensor with same shape as input.
    """
    return dispatch_amp("cumsum", _cumsum_impl, a, axis=axis)

def _mean_impl(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = ensure_tensor(a)
    data = xp.mean(a.data, axis=axis, keepdims=keepdims)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "mean"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        if axis is not None and not keepdims:
            grad_out = xp.expand_dims(grad_out, axis=axis)
        div = xp.prod([a.shape[ax] for ax in (axis if isinstance(axis, tuple) else ([axis] if axis is not None else range(a.data.ndim)))])
        grad_a = xp.broadcast_to(grad_out / div, a.shape)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def mean(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """
    Reduction mean: out = mean(a)

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axis/axes to reduce.
        keepdims (bool): Whether to retain reduced dims.

    Returns:
        Tensor: Mean tensor.
    """
    return dispatch_amp("mean", _mean_impl, a, axis=axis, keepdims=keepdims)

def _max_impl(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = ensure_tensor(a)
    data = xp.max(a.data, axis=axis, keepdims=keepdims)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "max"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        if axis is not None and not keepdims:
            grad_out = xp.expand_dims(grad_out, axis=axis)
        mask = (a.data == xp.expand_dims(data, axis=axis)) if axis is not None else (a.data == data)
        grad_a = mask.astype(a.dtype) * xp.broadcast_to(grad_out, a.shape)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def max(a: Tensor, axis=None, keepdims=False):
    """
    Reduction max: out = max(a)

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axis/axes to reduce.
        keepdims (bool): Whether to retain reduced dims.

    Returns:
        Tensor: Maximum values.
    """
    return dispatch_amp("max", _max_impl, a, axis=axis, keepdims=keepdims)

def _min_impl(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = ensure_tensor(a)
    data = xp.min(a.data, axis=axis, keepdims=keepdims)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "min"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        if axis is not None and not keepdims:
            grad_out = xp.expand_dims(grad_out, axis=axis)
        mask = (a.data == xp.expand_dims(data, axis=axis)) if axis is not None else (a.data == data)
        grad_a = mask.astype(a.dtype) * xp.broadcast_to(grad_out, a.shape)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def min(a: Tensor, axis=None, keepdims=False):
    """
    Reduction min: out = min(a)

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axis/axes to reduce.
        keepdims (bool): Whether to retain reduced dims.

    Returns:
        Tensor: Minimum values.
    """
    return dispatch_amp("min", _min_impl, a, axis=axis, keepdims=keepdims)

def _argmax_impl(a: Tensor, axis=None) -> Tensor:
    a = ensure_tensor(a)
    data = xp.argmax(a.data, axis=axis)
    out = Tensor(data, requires_grad=False, dtype=a.dtype)  # non-diff
    out.is_leaf = False
    out.grad_fn = "argmax"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    return out

def argmax(a: Tensor, axis=None) -> Tensor:
    """
    Returns the indices of the maximum values along an axis.

    Args:
        a (Tensor): Input tensor.
        axis (int, optional): Axis along which to find the maximum. 
                              If None, returns the index of the global maximum.

    Returns:
        Tensor: Tensor of indices (non-differentiable).

    Notes:
        - This operation does not support gradients.
        - `requires_grad=False` is always enforced.
    """
    return dispatch_amp("argmax", _argmax_impl, a, axis=axis)

def _argmin_impl(a: Tensor, axis=None) -> Tensor:
    a = ensure_tensor(a)
    data = xp.argmin(a.data, axis=axis)
    out = Tensor(data, requires_grad=False, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "argmin"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    return out

def argmin(a: Tensor, axis=None) -> Tensor:
    return dispatch_amp("argmin", _argmin_impl, a, axis=axis)

def _var_impl(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = ensure_tensor(a)
    mean_a = mean(a, axis=axis, keepdims=True)   # use autograd mean
    diff = a - mean_a
    sq = diff * diff
    out = mean(sq, axis=axis, keepdims=keepdims)
    out.is_leaf = False
    out.grad_fn = "var"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    return out

def var(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """
    Variance: out = var(a)

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axes to reduce.
        keepdims (bool): Whether to retain reduced dimensions.
        eps (float): Numerical stability.

    Returns:
        Tensor: Variance along given axes.
    """
    return dispatch_amp("var", _var_impl, a, axis=axis, keepdims=keepdims)

def _std_impl(a: Tensor, axis=None, keepdims=False, eps=1e-8) -> Tensor:
    a = ensure_tensor(a)
    v = var(a, axis=axis, keepdims=keepdims)
    out = (v + eps) ** 0.5
    out.is_leaf = False
    out.grad_fn = "std"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    return out

def std(a: Tensor, axis=None, keepdims=False, eps=1e-8) -> Tensor:
    """
    Standard deviation: out = std(a)

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axes to reduce.
        keepdims (bool): Whether to retain reduced dimensions.
        eps (float): Numerical stability.

    Returns:
        Tensor: Standard deviation along given axes.
    """
    return dispatch_amp("std", _std_impl, a, axis=axis, keepdims=keepdims, eps=eps)

def _logsumexp_impl(a: Tensor, axis=None, keepdims=False) -> Tensor:
    a = ensure_tensor(a)
    max_a = max(a, axis=axis, keepdims=True)
    shifted = a - max_a

    exp_shifted = exp(shifted)
    sum_exp = sum(exp_shifted, axis=axis, keepdims=keepdims)

    log_sum_exp = log(sum_exp)

    if keepdims:
        out = max_a + log_sum_exp
    else:
        out = squeeze(max_a, axis=axis) + log_sum_exp
    
    out.is_leaf = False
    out.grad_fn = "logsumexp"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    return out

def logsumexp(a: Tensor, axis=None, keepdims=False) -> Tensor:
    """
    Log-Sum-Exp trick: log(sum(exp(a)))

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axes to reduce.
        keepdims (bool): Whether to retain reduced dims.

    Returns:
        Tensor: log-sum-exp values.
    """
    return dispatch_amp("logsumexp", _logsumexp_impl, a, axis=axis, keepdims=keepdims)

# ============================================================
# Matrix operations
# ============================================================

def _matmul_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    dtype = promote_dtype(a.dtype, b.dtype)
    data = xp.matmul(a.data.astype(dtype), b.data.astype(dtype))
    requires_grad = a.requires_grad or b.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "matmul"

    for t in (a, b):
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        if not (a.requires_grad or b.requires_grad):
            return
        
        grad_out = out.grad.astype(dtype, copy=False)
        if a.requires_grad:
            grad_a = xp.matmul(grad_out, b.data.T.astype(dtype))
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad += grad_a

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

        if b.requires_grad:
            grad_b = xp.matmul(a.data.T.astype(dtype), grad_out)
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad += grad_b

            for hook in getattr(b, "_grad_hooks", []):
                new_grad = hook(b.grad)
                if new_grad is not None:
                    b.grad = new_grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Matrix multiplication of two tensors.

    Args:
        a (Tensor): Left-hand side tensor.
        b (Tensor): Right-hand side tensor.

    Returns:
        Tensor: Result of matrix multiplication.

    Notes:
        - Supports autograd.
        - Gradient w.r.t. `a`: grad_out @ b^T
        - Gradient w.r.t. `b`: a^T @ grad_out
    """
    return dispatch_amp("matmul", _matmul_impl, a, b)

def _dot_impl(a: Tensor, b: Tensor) -> Tensor:
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    dtype = promote_dtype(a.dtype, b.dtype)
    data = xp.dot(a.data.astype(dtype), b.data.astype(dtype))
    requires_grad = a.requires_grad or b.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "dot"

    for t in (a, b):
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        if not (a.requires_grad or b.requires_grad):
            return
        
        grad_out = out.grad.astype(dtype, copy=False)

        if a.requires_grad:
            grad_a = grad_out * b.data
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad += grad_a

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

        if b.requires_grad:
            grad_b = grad_out * a.data
            if b.grad is None:
                b.grad = grad_b
            else:
                b.grad += grad_b

            for hook in getattr(b, "_grad_hooks", []):
                new_grad = hook(b.grad)
                if new_grad is not None:
                    b.grad = new_grad

    out._backward = _backward
    out._prev = {a, b}
    return out

def dot(a: Tensor, b: Tensor) -> Tensor:
    """
    Dot product of two tensors.

    Args:
        a (Tensor): First input tensor.
        b (Tensor): Second input tensor.

    Returns:
        Tensor: Scalar or tensor result of dot product.

    Notes:
        - Gradient w.r.t. `a`: grad_out * b
        - Gradient w.r.t. `b`: grad_out * a
    """
    return dispatch_amp("dot", _dot_impl, a, b)

def _transpose_impl(a: Tensor, axes=None) -> Tensor:
    a = ensure_tensor(a)
    data = xp.transpose(a.data, axes=axes)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "transpose"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        if axes is None:
            grad_a = xp.transpose(out.grad, axes=axes)
        else:
            inv_axes = xp.argsort(axes)
            grad_a = xp.transpose(out.grad, axes=inv_axes)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def transpose(a: Tensor, axes=None) -> Tensor:
    """
    Transpose tensor (permute axes).

    Args:
        a (Tensor): Input tensor.
        axes (tuple, optional): Axis permutation.

    Returns:
        Tensor: Transposed tensor.
    """
    return dispatch_amp("transpose", _transpose_impl, a, axes=axes)

def _reshape_impl(a: Tensor, new_shape) -> Tensor:
    a = ensure_tensor(a)
    data = a.data.reshape(new_shape)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "reshape"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_a = out.grad.reshape(a.shape)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def reshape(a: Tensor, new_shape) -> Tensor:
    """
    Reshape tensor.

    Args:
        a (Tensor): Input tensor.
        shape (tuple): New shape.

    Returns:
        Tensor: Reshaped tensor.
    """
    return dispatch_amp("reshape", _reshape_impl, a, new_shape)

def _pad_impl(a: Tensor, pad_width, mode='constant', constant_values=0) -> Tensor:
    a = ensure_tensor(a)
    data = xp.pad(a.data, pad_width, mode=mode, constant_values=constant_values)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "pad"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        # extract the original slice to propagate grad
        slices = tuple(slice(p[0], out.shape[i]-p[1]) for i, p in enumerate(pad_width))
        grad_a = out.grad[slices]
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def pad(a: Tensor, pad_width, mode="constant", constant_values=0) -> Tensor:
    """
    Pads a tensor.

    Args:
        a (Tensor): Input tensor.
        pad_width (tuple): Number of values padded to edges of each axis.
        mode (str): Padding mode (e.g., 'constant', 'reflect', etc.).
        constant_values (scalar): Constant value if mode='constant'.

    Returns:
        Tensor: Padded tensor.

    Notes:
        - Gradient is propagated by slicing the padded result back to the original shape.
    """
    return dispatch_amp("pad", _pad_impl, a, pad_width, mode=mode, constant_values=constant_values)

def _concatenate_impl(tensors: list[Tensor], axis=0) -> Tensor:
    tensors = [ensure_tensor(t) for t in tensors]
    dtype = promote_dtype(*tensors)
    datas = [t.data.astype(dtype, copy=False) for t in tensors]
    data = xp.concatenate(datas, axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=dtype)
    out.is_leaf = False
    out.grad_fn = "concatenate"

    for t in tensors:
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        if not any(t.requires_grad for t in tensors):
            return
        
        splits = xp.cumsum([t.shape[axis] for t in tensors[:-1]])
        grads = xp.split(out.grad, splits, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                if t.grad is None:
                    t.grad = g
                else:
                    t.grad += g

                for hook in getattr(t, "_grad_hooks", []):
                    new_grad = hook(t.grad)
                    if new_grad is not None:
                        t.grad = new_grad

    out._backward = _backward
    out._prev = set(tensors)
    return out

def concatenate(tensors: list[Tensor], axis=0) -> Tensor:
    """
    Concatenates a sequence of tensors along a given axis.

    Args:
        tensors (list[Tensor]): List of tensors to concatenate.
        axis (int): Axis along which to concatenate.

    Returns:
        Tensor: Concatenated tensor.

    Notes:
        - Gradient is propagated by splitting the upstream gradient 
          back into chunks matching each input tensor.
    """
    return dispatch_amp("concatenate", _concatenate_impl, tensors, axis=axis)

def _stack_impl(tensors: list[Tensor], axis=0) -> Tensor:
    if not tensors:
        raise ValueError("ops.stack() requires a non-empty list of tensors.")

    # Ensure all are Tensor
    tensors = [ensure_tensor(t) for t in tensors]

    data = xp.stack([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    if not backend.is_grad_enabled():
        requires_grad = False

    out = Tensor(data, requires_grad=requires_grad, dtype=tensors[0].dtype)
    out.is_leaf = False
    out.grad_fn = "stack"

    for t in tensors:
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        if not any(t.requires_grad for t in tensors):
            return
        
        grad_out = out.grad

        # Split gradient along the stacking axis
        grads_split = xp.split(grad_out, len(tensors), axis=axis)

        for t, g in zip(tensors, grads_split):
            if not t.requires_grad:
                continue
            if t.grad is None:
                t.grad = g
            else:
                t.grad += g

            for hook in getattr(t, "_grad_hooks", []):
                new_grad = hook(t.grad)
                if new_grad is not None:
                    t.grad = new_grad

    out._backward = _backward
    out._prev = set(tensors)

    return out

def stack(tensors: list[Tensor], axis=0) -> Tensor:
    """
    Stack a sequence of tensors along a new axis.

    Args:
        tensors (List[Tensor]): Sequence of Tensor objects with identical shapes.
        axis (int): Axis to insert the new dimension along.

    Returns:
        Tensor: Stacked output tensor with autograd support.
    """
    return dispatch_amp("stack", _stack_impl, tensors, axis=axis)

def _split_impl(a: Tensor, sections, axis=0):
    a = ensure_tensor(a)
    data_splits = xp.array_split(a.data, sections, axis=axis)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False

    outs = [Tensor(split, requires_grad=requires_grad, dtype=a.dtype) for split in data_splits]
    for t in outs:
        t.is_leaf = False
        t.grad_fn = "split"

        for hook in getattr(a, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if not a.requires_grad:
            return
        if any(t.grad is not None for t in outs):
            grads = [t.grad if t.grad is not None else xp.zeros_like(t.data) for t in outs]
            grad_concat = xp.concatenate(grads, axis=axis)
            if a.grad is None:
                a.grad = grad_concat
            else:
                a.grad += grad_concat

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

    for t in outs:
        t._backward = _backward
        t._prev = {a}

    return outs

def split(a: Tensor, sections, axis=0):
    """
    Split a tensor into multiple sub-tensors along a given axis.

    Args:
        a (Tensor): Input tensor to split.
        sections (int or sequence): If int, number of equal splits.
                                    If sequence, explicit split indices.
        axis (int): Axis to split along.

    Returns:
        List[Tensor]: A list of views (sub-tensors) along the specified axis.
    """
    return dispatch_amp("split", _split_impl, a, sections, axis=axis)

def _squeeze_impl(a: Tensor, axis=None) -> Tensor:
    a = ensure_tensor(a)
    data = xp.squeeze(a.data, axis=axis)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "squeeze"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        # unsqueeze along the same axis to restore original shape
        if axis is None:
            target_shape = a.shape
        else:
            if isinstance(axis, int):
                target_shape = list(data.shape)
                target_shape.insert(axis, 1)
            else:
                # multiple axes: just restore to original shape
                target_shape = a.shape
        grad_a = grad_out.reshape(target_shape)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def squeeze(a: Tensor, axis=None) -> Tensor:
    """
    Remove single-dimensional entries from the shape of a tensor.

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple of ints, optional): 
            Selects a subset of dimensions to squeeze. 
            If None, all dimensions of size 1 will be removed.

    Returns:
        Tensor: Squeezed tensor.
    """
    return dispatch_amp("squeeze", _squeeze_impl, a, axis=axis)

def _unsqueeze_impl(a: Tensor, axis: int) -> Tensor:
    a = ensure_tensor(a)
    data = xp.expand_dims(a.data, axis=axis)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "unsqueeze"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        # reduce (sum) along the added axis
        grad_a = grad_out.sum(axis=axis)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def unsqueeze(a: Tensor, axis: int) -> Tensor:
    """
    Insert a new axis of length one at the specified position.

    Args:
        a (Tensor): Input tensor.
        axis (int): Position where a new axis will be inserted.

    Returns:
        Tensor: Tensor with an added dimension of size 1.
    """
    return dispatch_amp("unsqueeze", _unsqueeze_impl, a, axis)

def _slice_impl(a: Tensor, slices) -> Tensor:
    a = ensure_tensor(a)
    data = a.data[slices]  # forward
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "slice"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_a = xp.zeros_like(a.data)
        grad_a[slices] = out.grad
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def slice(a: Tensor, slices) -> Tensor:
    """
    Slice a tensor to extract a subset of elements.

    Args:
        a (Tensor): Input tensor.
        slices (slice or tuple of slices): Indexing specification.

    Returns:
        Tensor: Tensor containing the selected elements.
    """
    return dispatch_amp("slice", _slice_impl, a, slices)

def _flip_impl(a: Tensor, axis=None) -> Tensor:
    a = ensure_tensor(a)
    data = xp.flip(a.data, axis=axis)
    requires_grad = a.requires_grad 
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "flip"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_input = xp.flip(out.grad, axis=axis)
        if a.grad is None:
            a.grad = grad_input
        else:
            a.grad += grad_input

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def flip(a: Tensor, axis=None) -> Tensor:
    """
    Reverse the order of elements along specified axes.

    Args:
        a (Tensor): Input tensor.
        axes (Union[int, Tuple[int, ...]]): Axis or axes to flip. Default is all axes.

    Returns:
        Tensor: Tensor with elements reversed along the specified axes.
    """
    return dispatch_amp("flip", _flip_impl, a, axis=axis)

def _roll_impl(a: Tensor, shift, axis=None) -> Tensor:
    a = ensure_tensor(a)
    data = xp.roll(a.data, shift, axis=axis)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "roll"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_input = xp.roll(out.grad, -shift, axis=axis)
        if a.grad is None:
            a.grad = grad_input
        else:
            a.grad += grad_input

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def roll(a: Tensor, shift, axis=None) -> Tensor:
    """
    Roll (circularly shift) tensor elements along specified axes.

    Args:
        a (Tensor): Input tensor.
        shift (int or tuple of ints): Number of positions to shift. Positive values shift right.
        axis (int or tuple of ints): Axis or axes along which to roll. Default: flattened tensor.

    Returns:
        Tensor: Tensor with the same shape as `a`, with elements rolled along the specified axes.
    """
    return dispatch_amp("roll", _roll_impl, a, shift, axis=axis)

def _inv_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    data = xp.linalg.inv(a.data)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "inv"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return

        grad_out = out.grad
        grad_a = -xp.matmul(out.data.T, xp.matmul(grad_out, out.data.T))
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def inv(a: Tensor) -> Tensor:
    """
    Compute the inverse of a square matrix tensor.

    Args:
        a (Tensor): Square input tensor.

    Returns:
        Tensor: Inverse of the input tensor.
    """
    return dispatch_amp("inv", _inv_impl, a)

def _det_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    data = xp.linalg.det(a.data)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "det"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        inv_a_T = xp.linalg.inv(a.data).T
        grad_a = grad_out * out.data * inv_a_T
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def det(a: Tensor) -> Tensor:
    """
    Compute the determinant of a square matrix tensor.

    Args:
        a (Tensor): Square input tensor.

    Returns:
        Tensor: Scalar tensor containing the determinant.
    """
    return dispatch_amp("det", _det_impl, a)

def _trace_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    data = xp.trace(a.data)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "trace"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_a = out.grad * xp.eye(a.shape[0], dtype=a.dtype)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def trace(a: Tensor) -> Tensor:
    """
    Compute the trace (sum of diagonal elements) of a square matrix tensor.

    Args:
        a (Tensor): Square input tensor.

    Returns:
        Tensor: Scalar tensor containing the trace.
    """
    return dispatch_amp("trace", _trace_impl, a)

def _einsum_impl(subscripts: str, *operands: Tensor) -> Tensor:
    operands = [ensure_tensor(t) for t in operands]
    dtype = promote_dtype(*operands)
    datas = [t.data.astype(dtype, copy=False) for t in operands]
    data = xp.einsum(subscripts, *datas)
    requires_grad = any(t.requires_grad for t in operands)
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=dtype)
    out.is_leaf = False
    out.grad_fn = "einsum"

    for t in operands:
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        
        for i, t in enumerate(operands):
            if t.requires_grad:
                grad_t = xp.einsum(
                    subscripts.replace("->", f",{i}->"), 
                    out.grad, *(op.data for j, op in enumerate(operands) if j != i)
                )
                if t.grad is None:
                    t.grad = grad_t
                else:
                    t.grad += grad_t

                for hook in getattr(t, "_grad_hooks", []):
                    new_grad = hook(t.grad)
                    if new_grad is not None:
                        t.grad = new_grad

    out._backward = _backward
    out._prev = set(operands)
    return out

def einsum(subscripts: str, *operands: Tensor) -> Tensor:
    """
    Compute a generalized Einstein summation on the provided tensors.

    Args:
        subscripts (str): Subscript string specifying summation, e.g., 'ij,jk->ik'.
        *operands (Tensor): One or more tensors to participate in the summation.

    Returns:
        Tensor: Result of the Einstein summation.
    """
    return dispatch_amp("einsum", _einsum_impl, subscripts, *operands)

def _svd_impl(a: Tensor, full_matrices=False) -> Tensor:
    a = ensure_tensor(a)
    U, S, Vh = xp.linalg.svd(a.data, full_matrices=full_matrices)
    requires_grad = a.requires_grad 
    if not backend.is_grad_enabled():
        requires_grad = False
    U = Tensor(U, requires_grad=requires_grad, dtype=a.dtype)
    S = Tensor(S, requires_grad=requires_grad, dtype=a.dtype)
    Vh = Tensor(Vh, requires_grad=requires_grad, dtype=a.dtype)

    # Mark these as graph nodes
    for t in (U, S, Vh):
        t.is_leaf = False
        t.grad_fn = "svd"
        t._prev = {a}

        for hook in getattr(a, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        # SVD backward is non-trivial; here we use a simple pseudo-gradient approximation.
        # You can improve this with a full analytic Jacobian later if needed.
        if not a.requires_grad:
            return
        
        if any(t.grad is not None for t in (U, S, Vh)):
            grad_a = xp.zeros_like(a.data)
            # Approximate reconstruction gradient if available
            if S.grad is not None:
                grad_a += xp.matmul(U.data, xp.matmul(xp.diag(S.grad), Vh.data))
            if a.grad is None:
                a.grad = grad_a
            else:
                a.grad += grad_a

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

    # Attach same backward to each tensor
    U._backward = _backward
    S._backward = _backward
    Vh._backward = _backward

    return U, S, Vh

def svd(a: Tensor, full_matrices=False) -> Tensor:
    """
    Singular Value Decomposition (SVD).
    Decomposes a tensor into U, S, Vh such that: a = U @ diag(S) @ Vh

    Args:
        a (Tensor): Input tensor.
        full_matrices (bool): Whether to compute full-sized U, Vh (default: False).

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (U, S, Vh)
    """
    return dispatch_amp("svd", _svd_impl, a, full_matrices=full_matrices)

# ============================================================
# Tensor transformation/manipulation operations
# ============================================================

def _repeat_impl(a: Tensor, repeats, axis=None) -> Tensor:
    a = ensure_tensor(a)
    data = xp.repeat(a.data, repeats, axis=axis)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "repeat"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        # sum over repeated elements to route grad back
        if axis is None:
            grad_a = grad_out.reshape(a.shape[0], -1).sum(axis=1)
        else:
            grad_a = xp.add.reduceat(grad_out, xp.arange(0, grad_out.shape[axis], repeats), axis=axis)
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def repeat(a: Tensor, repeats, axis=None) -> Tensor:
    """
    Repeat elements of a tensor along a specified axis.

    Args:
        a (Tensor): Input tensor.
        repeats (int): Number of repetitions for each element.
        axis (int, optional): Axis along which to repeat values. If None, tensor is flattened.

    Returns:
        Tensor: Tensor with repeated elements.
    """
    return dispatch_amp("repeat", _repeat_impl, a, repeats, axis=axis)

def _tile_impl(a: Tensor, reps: tuple) -> Tensor:
    a = ensure_tensor(a)
    data = xp.tile(a.data, reps)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "tile"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        grad_in = grad_out.reshape(reps + a.shape).sum(axis=tuple(range(len(reps))))
        if a.grad is None:
            a.grad = grad_in
        else:
            a.grad += grad_in

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def tile(a: Tensor, reps: tuple) -> Tensor:
    """
    Repeat a tensor along each axis according to `reps`.

    Args:
        a (Tensor): Input tensor.
        reps (tuple of int): Number of repetitions along each axis.

    Returns:
        Tensor: Tensor with repeated elements.
    """
    return dispatch_amp("tile", _tile_impl, a, reps)

def _tril_impl(a: Tensor, k: int = 0) -> Tensor:
    """
    Lower-triangular part of a matrix.

    Args:
        a (Tensor): Input tensor (2D or batched 2D).
        k (int): Diagonal offset. 0 is main diagonal, 
                 k > 0 keeps k-th upper diagonals,
                 k < 0 keeps k-th lower diagonals.
    """
    a = ensure_tensor(a)
    data = xp.tril(a.data, k=k)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False

    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "tril"

    # Activation hooks
    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None or not a.requires_grad:
            return

        # Gradients only pass through the kept elements
        grad_mask = xp.tril(xp.ones_like(a.data), k=k)
        grad_in = out.grad * grad_mask

        if a.grad is None:
            a.grad = grad_in
        else:
            a.grad += grad_in

        # Grad hooks
        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def tril(a: Tensor, k: int = 0) -> Tensor:
    """
    Extract the lower-triangular part of a matrix (zeroing others).

    Args:
        a (Tensor): Input tensor (2D or batched 2D).
        k (int, optional): Diagonal offset. Defaults to 0.

    Returns:
        Tensor: Lower-triangular view of the input.
    """
    return dispatch_amp("tril", _tril_impl, a, k=k)

def _expand_impl(a: Tensor, shape: tuple) -> Tensor:
    a = ensure_tensor(a)
    data = xp.broadcast_to(a.data, shape)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "expand"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        grad_out = unbroadcast(grad_out, a.shape)
        if a.grad is None:
            a.grad = grad_out
        else:
            a.grad += grad_out

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def expand(a: Tensor, shape: tuple) -> Tensor:
    """
    Broadcast a tensor to a new shape.

    Args:
        a (Tensor): Input tensor.
        shape (tuple of int): Target shape to broadcast to.

    Returns:
        Tensor: Broadcasted tensor.
    """
    return dispatch_amp("expand", _expand_impl, a, shape)

def _normalize_impl(a: Tensor, axis=None, epsilon=1e-8) -> Tensor:
    a = ensure_tensor(a)
    requires_grad = a.requires_grad and backend.is_grad_enabled()

    norm = xp.sqrt(xp.sum(a.data ** 2, axis=axis, keepdims=True)) + epsilon
    data = a.data / norm

    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "normalize"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad
        # gradient of normalization: 
        # da = (grad_out - a * sum(a * grad_out) / norm^2) / norm
        norm_val = norm
        grad_a = (grad_out - a.data * xp.sum(a.data * grad_out, axis=axis, keepdims=True) / (norm_val ** 2)) / norm_val
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def normalize(a: Tensor, axis=None, epsilon=1e-8):
    """
    Normalize tensor along specified axis.

    Equivalent to: a / (sqrt(sum(a^2, axis)) + epsilon)

    Args:
        a (Tensor): Input tensor.
        axis (int or tuple, optional): Axis or axes to normalize across.
        epsilon (float, optional): Small constant for numerical stability.

    Returns:
        Tensor: Normalized tensor.
    """
    return dispatch_amp("normalize", _normalize_impl, a, axis=axis, epsilon=epsilon)

def _normalize_absmax_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    abs_block = abs(a)
    scale = max(abs_block)
    scale = where(scale == 0, Tensor(1.0), scale)
    return divide(a, scale)

def normalize_absmax(a: Tensor) -> Tensor:
    """
    Scale Tensor to [-1, 1] using absolute maximum.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: Normalized tensor.
    """
    return dispatch_amp("normalize_absmax", _normalize_absmax_impl, a)

def _renorm_impl(a: Tensor, p=2.0, dim=0, maxnorm=1.0) -> Tensor:
    a = ensure_tensor(a)
    requires_grad = a.requires_grad and backend.is_grad_enabled()

    # Compute p-norms along dimension
    norms = xp.linalg.norm(a.data, ord=p, axis=dim, keepdims=True) + 1e-8

    # Determine scaling factor
    scale = xp.clip(maxnorm / norms, 0.0, 1.0)

    # Apply scaling only where norm > maxnorm
    mask = (norms > maxnorm).astype(a.data.dtype)
    scaled = a.data * (scale * mask + (1.0 - mask))

    out = Tensor(scaled, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "renorm"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_out = out.grad

        # Compute gradient only for elements that were rescaled
        grad_a = grad_out * (scale * mask + (1.0 - mask))
        if a.grad is None:
            a.grad = grad_a
        else:
            a.grad += grad_a

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def renorm(a: Tensor, p=2.0, dim=0, maxnorm=1.0) -> Tensor:
    """
    Renormalize sub-tensors along a given dimension so their p-norm <= maxnorm.
    Commonly used for gradient clipping and max-norm regularization.

    Args:
        a (Tensor): Input tensor.
        p (float, optional): The p-norm type (default: 2.0).
        dim (int, optional): The dimension along which to compute norms.
        maxnorm (float, optional): Maximum allowed norm value.

    Returns:
        Tensor: Renormalized tensor with the same shape as input.
    """
    return dispatch_amp("renorm", _renorm_impl, a, p=p, dim=dim, maxnorm=maxnorm)

def _sort_impl(a: Tensor, dim: int = -1, descending: bool = False) -> tuple[Tensor, Tensor]:
    a = ensure_tensor(a)
    data = a.data
    idx = xp.argsort(data, axis=dim)
    if descending:
        idx = idx[..., ::-1]
    values = xp.take_along_axis(data, idx, axis=dim)

    values_tensor = Tensor(values, requires_grad=False, dtype=a.dtype)
    indices_tensor = Tensor(idx.astype(xp.int64), requires_grad=False, dtype=xp.int64)

    for hook in getattr(a, "_activation_hooks", []):
        new_values_tensor = hook(values_tensor)
        if new_values_tensor is not None:
            values_tensor = new_values_tensor

    values_tensor.is_leaf = False
    indices_tensor.is_leaf = False
    values_tensor.grad_fn = "sort_values"
    indices_tensor.grad_fn = "sort_indices"
    return values_tensor, indices_tensor

def sort(a: Tensor, dim: int = -1, descending: bool = False) -> tuple[Tensor, Tensor]:
    """
    Sort tensor along specified dimension.

    Returns sorted values and original indices.

    Args:
        x (Tensor): Input tensor.
        dim (int, optional): Axis to sort along. Default: -1.
        descending (bool): Sort in descending order if True.

    Returns:
        tuple[Tensor, Tensor]: (sorted_values, indices)
    """
    return dispatch_amp("sort", _sort_impl, a, dim=dim, descending=descending)

def _searchsorted_impl(sorted_sequence: Tensor, values: Tensor, side: str = 'left') -> Tensor:
    sorted_sequence = ensure_tensor(sorted_sequence)
    values = ensure_tensor(values)
    data_seq = sorted_sequence.data
    data_val = values.data
    idx = xp.searchsorted(data_seq, data_val, side=side)
    indices_tensor = Tensor(idx.astype(xp.int64), requires_grad=False, dtype=xp.int64)

    for hook in getattr(values, "_activation_hooks", []):  # Hooks on values, as it's the varying input
        new_indices_tensor = hook(indices_tensor)
        if new_indices_tensor is not None:
            indices_tensor = new_indices_tensor

    indices_tensor.is_leaf = False
    indices_tensor.grad_fn = "searchsorted"
    return indices_tensor

def searchsorted(sorted_sequence: Tensor, values: Tensor, side: str = "left") -> Tensor:
    """
    Find the indices where *values* would be inserted into *sorted_sequence*
    to maintain sorted order.

    Args:
        sorted_sequence (Tensor): **1-D** sorted tensor (ascending).  
            Multi-dimensional tensors are treated as flattened.
        values (Tensor): Values to search for. May be any shape; broadcasting
            follows NumPy rules.
        side (str, optional): ``'left'`` (default) returns the first suitable
            index, ``'right'`` returns the index after the last equal element.

    Returns:
        Tensor: Integer tensor of the same shape as *values* containing the
        insertion indices.
    """
    return dispatch_amp("searchsorted", _searchsorted_impl, sorted_sequence, values, side=side)

def _im2col_impl(X: Tensor, kernel_size: tuple, s: int) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col, col2im
    X = ensure_tensor(X)
    data = im2col(X.data, kernel_size, s)
    requires_grad = X.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=X.dtype)
    out.is_leaf = False
    out.grad_fn = "im2col"

    for hook in getattr(X, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not X.requires_grad:
            return
        
        # Gradient w.r.t. input X comes from col2im of upstream gradient
        dX = col2im(out.grad, X.shape, kernel_size, s)
        if X.grad is None:
            X.grad = dX
        else:
            X.grad += dX

        for hook in getattr(X, "_grad_hooks", []):
            new_grad = hook(X.grad)
            if new_grad is not None:
                X.grad = new_grad

    out._backward = _backward
    out._prev = {X}
    return out

def im2col(X: Tensor, kernel_size: tuple, s: int) -> Tensor:
    """
    Transform a 4D input tensor into column form for convolutions.

    Args:
        X (Tensor): Input tensor of shape (N, C, H, W).
        f (int): Filter size.
        s (int): Stride for the convolution.

    Returns:
        Tensor: Column-form tensor.
    """
    return dispatch_amp("im2col", _im2col_impl, X, kernel_size, s)

def _col2im_impl(cols: Tensor, X_shape, kernel_size: tuple, s: int) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col, col2im
    cols = ensure_tensor(cols)
    data = col2im(cols.data, X_shape, kernel_size, s)
    requires_grad = cols.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=cols.dtype)
    out.is_leaf = False
    out.grad_fn = "col2im"

    for hook in getattr(cols, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not cols.requires_grad:
            return
        
        # Gradient w.r.t. cols comes from im2col of upstream gradient
        dcols = im2col(out.grad, kernel_size, s)
        if cols.grad is None:
            cols.grad = dcols
        else:
            cols.grad += dcols

        for hook in getattr(cols, "_grad_hooks", []):
            new_grad = hook(cols.grad)
            if new_grad is not None:
                cols.grad = new_grad

    out._backward = _backward
    out._prev = {cols}
    return out

def col2im(cols: Tensor, X_shape, kernel_size: tuple, s: int) -> Tensor:
    """
    Transform a column-form tensor back to its original 4D image shape.

    Args:
        cols (Tensor): Column-form tensor.
        X_shape (tuple): Shape of the target output tensor (N, C, H, W).
        f (int): Filter size used in im2col.
        s (int): Stride used in im2col.

    Returns:
        Tensor: Reconstructed 4D tensor.
    """
    return dispatch_amp("col2im", _col2im_impl, cols, X_shape, kernel_size, s)

def _im2col_transpose_impl(X: Tensor, kernel_size: tuple, s: int, output_shape: tuple) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col_transpose, col2im_transpose
    X = ensure_tensor(X)
    data = im2col_transpose(X.data, kernel_size, s, output_shape)
    requires_grad = X.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=X.dtype)
    out.is_leaf = False
    out.grad_fn = "im2col_transpose"

    for hook in getattr(X, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not X.requires_grad:
            return
        
        # Gradient w.r.t. input X comes from col2im_transpose of upstream gradient
        dX = col2im_transpose(out.grad, X.shape, kernel_size, s, output_shape)
        if X.grad is None:
            X.grad = dX
        else:
            X.grad += dX

        for hook in getattr(X, "_grad_hooks", []):
            new_grad = hook(X.grad)
            if new_grad is not None:
                X.grad = new_grad

    out._backward = _backward
    out._prev = {X}
    return out

def im2col_transpose(X: Tensor, kernel_size: tuple, s: int, output_shape: tuple) -> Tensor:
    """
    Transform a 4D input tensor into column form for transposed convolutions.

    Args:
        X (Tensor): Input tensor of shape (N, C, H, W).
        f (int): Filter size.
        s (int): Stride for the transposed convolution.
        output_shape (tuple): Shape of the expected output tensor (N, C, H_out, W_out).

    Returns:
        Tensor: Column-form tensor.
    """
    return dispatch_amp("im2col_transpose", _im2col_transpose_impl, X, kernel_size, s, output_shape)

def _col2im_transpose_impl(cols: Tensor, X_shape: tuple, kernel_size: tuple, s: int, output_shape: tuple) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col_transpose, col2im_transpose
    cols = ensure_tensor(cols)
    data = col2im_transpose(cols.data, X_shape, kernel_size, s, output_shape)
    requires_grad = cols.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=cols.dtype)
    out.is_leaf = False
    out.grad_fn = "col2im_transpose"

    for hook in getattr(cols, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not cols.requires_grad:
            return
        
        # Gradient w.r.t. cols comes from im2col_transpose of upstream gradient
        dcols = im2col_transpose(out.grad, kernel_size, s, output_shape)
        if cols.grad is None:
            cols.grad = dcols
        else:
            cols.grad += dcols

        for hook in getattr(cols, "_grad_hooks", []):
            new_grad = hook(cols.grad)
            if new_grad is not None:
                cols.grad = new_grad

    out._backward = _backward
    out._prev = {cols}
    return out

def col2im_transpose(cols: Tensor, X_shape: tuple, kernel_size: tuple, s: int, output_shape: tuple) -> Tensor:
    """
    Transform a column-form tensor back into a 4D image tensor for transposed convolutions.

    Args:
        cols (Tensor): Column-form tensor.
        X_shape (tuple): Input tensor shape (N, C, H, W) before transposed convolution.
        f (int): Filter size used in im2col_transposed.
        s (int): Stride used in im2col_transposed.
        output_shape (tuple): Shape of the target output tensor (N, C, H_out, W_out).

    Returns:
        Tensor: Reconstructed 4D tensor after transposed convolution.
    """
    return dispatch_amp("col2im_transpose", _col2im_transpose_impl, cols, X_shape, kernel_size, s, output_shape)

def _im2col_grouped_impl(X: Tensor, kernel_size: tuple, s: int, groups: int) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col_grouped, col2im_grouped
    X = ensure_tensor(X)
    data = im2col_grouped(X.data, kernel_size, s, groups)
    requires_grad = X.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=X.dtype)
    out.is_leaf = False
    out.grad_fn = "im2col_grouped"

    for hook in getattr(X, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not X.requires_grad:
            return

        dX = col2im_grouped(out.grad, kernel_size, s, groups)
        if X.grad is None:
            X.grad = dX
        else:
            X.grad += dX

        for hook in getattr(X, "_grad_hooks", []):
            new_grad = hook(X.grad)
            if new_grad is not None:
                X.grad = new_grad

    out._backward = _backward
    out._prev = {X}
    return out

def im2col_grouped(X: Tensor, kernel_size: tuple, s: int, groups: int) -> Tensor:
    """
    Transform a 4D input tensor into column form for grouped convolutions.

    Args:
        X (Tensor): Input tensor of shape (N, C, H, W).
        f (int): Filter size.
        s (int): Convolution stride.
        groups (int): Number of groups to split the input channels into.
                      Must evenly divide C.

    Returns:
        Tensor: Column-form tensor suitable for grouped convolution.
    """
    return dispatch_amp("im2col_grouped", _im2col_grouped_impl, X, kernel_size, s, groups)

def _col2im_grouped_impl(cols: Tensor, kernel_size: tuple, s: int, groups: int) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col_grouped, col2im_grouped
    cols = ensure_tensor(cols)
    data = col2im_grouped(cols.data, kernel_size, s, groups)
    requires_grad = cols.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=cols.dtype)
    out.is_leaf = False
    out.grad_fn = "col2im_grouped"

    for hook in getattr(cols, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not cols.requires_grad:
            return
        
        dcols = im2col_grouped(out.grad, kernel_size, s, groups)
        if cols.grad is None:
            cols.grad = dcols
        else:
            cols.grad += dcols

        for hook in getattr(cols, "_grad_hooks", []):
            new_grad = hook(cols.grad)
            if new_grad is not None:
                cols.grad = new_grad

    out._backward = _backward
    out._prev = {cols}
    return out

def col2im_grouped(cols: Tensor, kernel_size: tuple, s: int, groups: int) -> Tensor:
    """
    Transform a column-form tensor back into a 4D image tensor for grouped convolutions.

    Args:
        cols (Tensor): Column-form tensor from grouped convolution.
        f (int): Filter size.
        s (int): Convolution stride.
        groups (int): Number of groups that were used in the forward pass.

    Returns:
        Tensor: Reconstructed 4D tensor of shape (N, C, H, W).
    """
    return dispatch_amp("col2im_grouped", _col2im_grouped_impl, cols, kernel_size, s, groups)

def _im2col_transpose_grouped_impl(X: Tensor, kernel_size: tuple, s: int, output_shape: tuple, groups: int) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col_transpose_grouped, col2im_transpose_grouped
    X = ensure_tensor(X)
    data = im2col_transpose_grouped(X.data, kernel_size, s, output_shape, groups)
    requires_grad = X.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=X.dtype)
    out.is_leaf = False
    out.grad_fn = "im2col_transpose_grouped"

    for hook in getattr(X, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not X.requires_grad:
            return
        
        dX = col2im_transpose_grouped(out.grad, X.shape, kernel_size, s, output_shape, groups)
        if X.grad is None:
            X.grad = dX
        else:
            X.grad += dX

        for hook in getattr(X, "_grad_hooks", []):
            new_grad = hook(X.grad)
            if new_grad is not None:
                X.grad = new_grad

    out._backward = _backward
    out._prev = {X}
    return out

def im2col_transpose_grouped(X: Tensor, kernel_size: tuple, s: int, output_shape: tuple, groups: int) -> Tensor:
    """
    Transform a 4D input tensor into column form for grouped transposed convolutions.

    This is the grouped version of `im2col_transpose`. It splits the input
    tensor into channel groups, applies the im2col operation for transposed
    convolution separately on each group, and concatenates the results. 
    Useful for implementing grouped transposed convolution (e.g. in ResNeXt-style
    deconvolution layers).

    Args:
        X (Tensor): Input tensor of shape (N, C_in, H_in, W_in).
        kernel_size (tuple): Filter size as (f_h, f_w).
        s (int): Stride used for the transposed convolution.
        output_shape (tuple): Target output shape (N, C_out, H_out, W_out).
        groups (int): Number of channel groups. `C_in` must be divisible by this value.

    Returns:
        Tensor: Column-form tensor for grouped transposed convolution.
    """
    return dispatch_amp("im2col_transpose_grouped", _im2col_transpose_grouped_impl, X, kernel_size, s, output_shape, groups)

def _col2im_transpose_grouped_impl(cols: Tensor, X_shape: tuple, kernel_size: tuple, s: int, output_shape: tuple, groups: int) -> Tensor:
    from LunarLearn.core.tensor.utils import im2col_transpose_grouped, col2im_transpose_grouped
    cols = ensure_tensor(cols)
    data = col2im_transpose_grouped(cols, X_shape, kernel_size, s, output_shape, groups)
    requires_grad = cols.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=cols.dtype)
    out.is_leaf = False
    out.grad_fn = "col2im_transpose_grouped"

    for hook in getattr(cols, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not cols.requires_grad:
            return
        
        dcols = im2col_transpose_grouped(out.grad, kernel_size, s, output_shape, groups)
        if cols.grad is None:
            cols.grad = dcols
        else:
            cols.grad += dcols

        for hook in getattr(cols, "_grad_hooks", []):
            new_grad = hook(cols.grad)
            if new_grad is not None:
                cols.grad = new_grad

    out._backward = _backward
    out._prev = {cols}
    return out

def col2im_transpose_grouped(cols: Tensor, X_shape: tuple, kernel_size: tuple, s: int, output_shape: tuple, groups: int) -> Tensor:
    """
    Reconstruct a 4D tensor from column form for grouped transposed convolutions.

    This is the grouped counterpart to `col2im_transpose`. It splits the column
    tensor into channel groups, applies the inverse col2im transpose operation
    to each group separately, and combines them into the final output tensor.
    This function is typically used in the backward pass of grouped transposed
    convolutions.

    Args:
        cols (Tensor): Column-form tensor from grouped transposed convolution.
        X_shape (tuple): Original input shape (N, C_in, H_in, W_in).
        kernel_size (tuple): Filter size as (f_h, f_w).
        s (int): Stride used for the transposed convolution.
        output_shape (tuple): Target output shape (N, C_out, H_out, W_out).
        groups (int): Number of channel groups. `C_in` must be divisible by this value.

    Returns:
        Tensor: Reconstructed 4D tensor of shape (N, C_out, H_out, W_out).
    """
    return dispatch_amp("col2im_transpose_grouped", _col2im_transpose_grouped_impl, cols, X_shape, kernel_size, s, output_shape, groups)

# ============================================================
# Indexing & Selection operations
# ============================================================

def _gather_impl(a: Tensor, dim: int, index: Tensor) -> Tensor:
    a = ensure_tensor(a)
    index = ensure_tensor(index)
    data = xp.take_along_axis(a.data, index.data, axis=dim)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "gather"

    for t in (a, index):
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if xp.__name__ == 'cupy':
            try:
                from cupyx.scatter_add import scatter_add
            except ImportError:
                from cupyx._scatter import scatter_add

        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_input = xp.zeros_like(a.data)
        if xp.__name__ == 'cupy':
            scatter_add(grad_input, xp.indices_like(index.data), out.grad)
        else:
            xp.add.at(grad_input, xp.indices_like(index.data), out.grad)
        if a.grad is None:
            a.grad = grad_input
        else:
            a.grad += grad_input

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def gather(a: Tensor, dim: int, index: Tensor) -> Tensor:
    """
    Gather elements from a tensor along an axis using index tensor.

    This operation is the inverse of `scatter` and is commonly used in loss calculations
    or attention mechanisms.

    Args:
        a (Tensor): Source tensor.
        axis (int): Axis along which to gather.
        index (Tensor): Indices of elements to gather. Must have the same shape as the output.

    Returns:
        Tensor: Gathered tensor with the same shape as `index`.
    """
    return dispatch_amp("gather", _gather_impl, a, dim, index)

def _scatter_impl(a: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    a = ensure_tensor(a)
    index = ensure_tensor(index)
    src = ensure_tensor(src)
    data = a.data.copy()
    xp.put_along_axis(data, index.data, src.data, axis=dim)
    requires_grad = (a.requires_grad or src.requires_grad)
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "scatter"

    for t in (a, index, src):
        for hook in getattr(t, "_activation_hooks", []):
            new_out = hook(out)
            if new_out is not None:
                out = new_out

    def _backward():
        if out.grad is None:
            return
        if not (a.requires_grad or src.requires_grad):
            return
        
        if a.requires_grad:
            if a.grad is None:
                a.grad = out.grad.copy()
            else:
                a.grad += out.grad

            for hook in getattr(a, "_grad_hooks", []):
                new_grad = hook(a.grad)
                if new_grad is not None:
                    a.grad = new_grad

        if src.requires_grad:
            grad_src = xp.take_along_axis(out.grad, index.data, axis=dim)
            if src.grad is None:
                src.grad = grad_src
            else:
                src.grad += grad_src

            for hook in getattr(src, "_grad_hooks", []):
                new_grad = hook(src.grad)
                if new_grad is not None:
                    src.grad = new_grad

    out._backward = _backward
    out._prev = {a, src}
    return out

def scatter(a: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor:
    """
    Scatter elements from a source tensor into a new tensor according to indices.

    This is the inverse of `gather` and is commonly used in gradient propagation
    or constructing one-hot encodings.

    Args:
        a (Tensor): Base tensor to scatter into.
        axis (int): Axis along which to scatter.
        index (Tensor): Indices at which to place elements from `src`.
        src (Tensor): Values to scatter. Must be broadcastable to `index` shape.

    Returns:
        Tensor: Tensor with elements from `src` scattered into `a` according to `index`.
    """
    return dispatch_amp("scatter", _scatter_impl, a, dim, index, src)

# ============================================================================
# Activations & Loss operations
# ============================================================================

def _tanh_impl(a: Tensor) -> Tensor:
    return unary_op(
        a,
        op=xp.tanh,
        grad_fn=lambda grad_out, a_data: grad_out * (1 - xp.tanh(a_data) ** 2),
        name="tanh",
    )

def tanh(a: Tensor) -> Tensor:
    """
    Elementwise hyperbolic tangent.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor: tanh(a).
    """
    return dispatch_amp("tanh", _tanh_impl, a)

def _relu_impl(a: Tensor) -> Tensor:
    out = maximum(0, a)
    out.grad_fn = "relu"
    return out

def relu(a: Tensor) -> Tensor:
    """
    Rectified Linear Unit (ReLU) activation function.
    Applies the element-wise function: ReLU(x) = max(0, x)

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor with the same shape as `x`, where all negative values are replaced with 0.
    """
    return dispatch_amp("relu", _relu_impl, a)

def _sigmoid_impl(a: Tensor) -> Tensor:
    a = ensure_tensor(a)
    sig = 1 / (1 + xp.exp(-a.data))
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(sig, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "sigmoid"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_input = out.grad * sig * (1 - sig)
        if a.grad is None:
            a.grad = grad_input
        else:
            a.grad += grad_input

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def sigmoid(a: Tensor) -> Tensor:
    """
    Sigmoid activation function.
    Applies the element-wise function: sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor with values in the range (0, 1).
    """
    return dispatch_amp("sigmoid", _sigmoid_impl, a)

def _softmax_impl(a: Tensor, axis=-1) -> Tensor:
    a = ensure_tensor(a)
    shifted = a.data - xp.max(a.data, axis=axis, keepdims=True)
    exp_x = xp.exp(shifted)
    data = exp_x / xp.sum(exp_x, axis=axis, keepdims=True)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "softmax"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        # Jacobian-vector product trick
        grad_input = out.data * (out.grad - xp.sum(out.grad * out.data, axis=axis, keepdims=True))
        if a.grad is None:
            a.grad = grad_input
        else:
            a.grad += grad_input

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def softmax(a: Tensor, axis=-1) -> Tensor:
    """
    Softmax activation function.
    Applies the function along a specified axis to convert logits into probabilities:

        softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    Args:
        x (Tensor): Input tensor (logits).
        axis (int): Axis along which to apply softmax (default: -1).

    Returns:
        Tensor: Probability tensor of the same shape as `x`, where values along `axis` sum to 1.
    """
    return dispatch_amp("softmax", _softmax_impl, a, axis=axis)

def _log_softmax_impl(a: Tensor, axis=-1) -> Tensor:
    a = ensure_tensor(a)
    shifted = a.data - xp.max(a.data, axis=axis, keepdims=True)
    log_sum_exp = xp.log(xp.sum(xp.exp(shifted), axis=axis, keepdims=True))
    data = shifted - log_sum_exp
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "log_softmax"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not a.requires_grad:
            return
        
        grad_input = out.grad - xp.exp(data) * xp.sum(out.grad, axis=axis, keepdims=True)
        if a.grad is None:
            a.grad = grad_input
        else:
            a.grad += grad_input

        for hook in getattr(a, "_grad_hooks", []):
            new_grad = hook(a.grad)
            if new_grad is not None:
                a.grad = new_grad

    out._backward = _backward
    out._prev = {a}
    return out

def log_softmax(a: Tensor) -> Tensor:
    """
    Log-Softmax activation function.
    Applies the log of softmax along a specified axis:

        log_softmax(x_i) = log(softmax(x_i))

    Args:
        x (Tensor): Input tensor (logits).
        axis (int): Axis along which to apply log-softmax (default: -1).

    Returns:
        Tensor: Log-probabilities of the same shape as `x`.
    """
    return dispatch_amp("log_softmax", _log_softmax_impl, a)

def _nll_loss_impl(log_probs: Tensor, target: Tensor) -> Tensor:
    log_probs = ensure_tensor(log_probs)
    target = ensure_tensor(target)
    data = -log_probs.data[xp.arange(target.shape[0]), target.data]
    requires_grad = log_probs.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data.mean(), requires_grad=requires_grad, dtype=log_probs.dtype)
    out.is_leaf = False
    out.grad_fn = "nll_loss"

    for hook in getattr(log_probs, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    def _backward():
        if out.grad is None:
            return
        if not log_probs.requires_grad:
            return
        
        grad = xp.zeros_like(log_probs.data)
        grad[xp.arange(target.shape[0]), target.data] = -1.0 / target.shape[0]
        if log_probs.grad is None:
            log_probs.grad = grad * out.grad
        else:
            log_probs.grad += grad * out.grad

        for hook in getattr(log_probs, "_grad_hooks", []):
            new_grad = hook(log_probs.grad)
            if new_grad is not None:
                log_probs.grad = new_grad

    out._backward = _backward
    out._prev = {log_probs}
    return out

def nll_loss(log_probs: Tensor, target: Tensor) -> Tensor:
    """
    Negative Log-Likelihood (NLL) loss.
    Computes the loss between log-probabilities and target class indices:

        loss = -log_probs[range(N), target]

    Args:
        log_probs (Tensor): Log-probabilities from a log-softmax layer of shape (N, C, ...).
        target (Tensor): Ground-truth class indices of shape (N, ...).

    Returns:
        Tensor: Scalar tensor representing the mean negative log-likelihood loss.
    """
    return dispatch_amp("nll_loss", _nll_loss_impl, log_probs, target)

def _cross_entropy_impl(a: Tensor, target: Tensor) -> Tensor:
    return nll_loss(log_softmax(a), target)

def cross_entropy(a: Tensor, target: Tensor) -> Tensor:
    """
    Cross-Entropy loss.
    Combines `log_softmax` and `nll_loss` in one step:

        loss = -log( exp(logits[target]) / sum_j exp(logits[j]) )

    Args:
        logits (Tensor): Raw, unnormalized model predictions of shape (N, C, ...).
        target (Tensor): Ground-truth class indices of shape (N, ...).

    Returns:
        Tensor: Scalar tensor representing the mean cross-entropy loss.
    """
    return dispatch_amp("cross_entropy", _cross_entropy_impl, a, target)

def _dropout_impl(a: Tensor, keep_prob: float, training: bool = True) -> Tensor:
    a = ensure_tensor(a)
    if not training or keep_prob >= 1.0:
        return a
    if keep_prob <= 0.0:
        return zeros(a.shape)
    
    mask = (xp.random.rand(*a.shape) < keep_prob).astype(a.dtype)
    scale = 1.0 / keep_prob
    out = a * mask * scale
    out.grad_fn = "dropout"
    return out

def dropout(a: Tensor, keep_prob: float, training: bool = True) -> Tensor:
    """
    Dropout operation.

    During training, randomly zeroes some elements of the input tensor with probability `1 - keep_prob`
    using samples from a Bernoulli distribution. The elements are then scaled by `1/keep_prob`
    to preserve expected value.

    During evaluation, returns the input unchanged.

    Args:
        a (Tensor): Input tensor.
        keep_prob (float): Probability of keeping an element. Must be in (0, 1].
        training (bool): If True, apply dropout. If False, return input as-is.

    Returns:
        Tensor: Output tensor with dropout applied (training) or unchanged (eval).
    """
    return dispatch_amp("dropout", _dropout_impl, a, keep_prob=keep_prob, training=training)

def _avg_pool2d_impl(a: Tensor, pool_size: int, strides: int, padding: int = 0) -> Tensor:
    m, n_C_prev, n_H_prev, n_W_prev = a.shape 
    n_C = n_C_prev
    n_H = int((n_H_prev + 2 * padding - pool_size) / strides) + 1
    n_W = int((n_W_prev + 2 * padding - pool_size) / strides) + 1

    a_pad = pad(a, ((0,0),(0,0),(padding,padding),(padding,padding)),
                mode='constant')

    cols = im2col(a_pad, pool_size, strides)
    cols_reshaped = cols.reshape(pool_size * pool_size, n_C, -1)

    out = mean(cols_reshaped, axis=0)  # shape: (C, N*out_h*out_w)
    out = out.reshape(n_C, n_H, n_W, m).transpose(3, 0, 1, 2)  # (N, C, H, W)
    out.grad_fn = "avg_pool2d"

def avg_pool2d(a: Tensor, pool_size: int, strides: int, padding: int) -> Tensor:
    return dispatch_amp("avg_pool2d", _avg_pool2d_impl, a, pool_size, strides, padding)

def _upsample_impl(a: Tensor, scale_factor: int = 2, mode: str = "bilinear", align_corners: bool = True) -> Tensor:
    from LunarLearn.core.tensor.utils import upsample
    a = ensure_tensor(a)
    if a.ndim != 4:
        raise ValueError("upsample expects 4D tensor (B, C, H, W)")

    B, C, H, W = a.shape
    new_H, new_W = H * scale_factor, W * scale_factor

    data = upsample(a.data, scale_factor, mode, align_corners)
    requires_grad = a.requires_grad
    if not backend.is_grad_enabled():
        requires_grad = False
    out = Tensor(data, requires_grad=requires_grad, dtype=a.dtype)
    out.is_leaf = False
    out.grad_fn = "upsample"

    for hook in getattr(a, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    out._upsample_config = {
        "scale_factor": scale_factor,
        "mode": mode,
        "align_corners": align_corners,
        "in_shape": a.shape
    }

    def _backward():
        if out.grad is None or not a.requires_grad:
            return

        grad = out.grad
        scale = scale_factor
        in_H, in_W = a.shape[2], a.shape[3]

        if mode == "nearest":
            # Sum repeated gradients
            grad = grad.reshape(B, C, in_H, scale, in_W, scale)
            grad = grad.sum(axis=3).sum(axis=5)  # sum over repeated dims
        elif mode == "bilinear":
            # Downsample grad via average pooling
            grad = avg_pool2d(grad, kernel_size=scale, stride=scale, padding=0)

        if a.grad is None:
            a.grad = grad
        else:
            a.grad += grad

    out._backward = _backward
    out._prev = {a}
    return out

def upsample(a: Tensor, scale_factor: int = 2, mode: str = "bilinear", align_corners: bool = True) -> Tensor:
    return dispatch_amp("upsample", _upsample_impl, a, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

def _multinomial_impl(probs: Tensor, num_samples: int = 1) -> Tensor:
    probs = ensure_tensor(probs)
    if probs.ndim != 2:
        raise ValueError("probs must be 2D (B, V)")
    if not xp.allclose(probs.sum(axis=-1), 1.0):
        probs = softmax(probs, axis=-1)

    B, V = probs.shape
    flat_probs = probs.data
    samples = xp.random.choice(
        V,
        size=(B, num_samples),
        p=flat_probs,
        replace=True
    )
    out = Tensor(samples, requires_grad=False, dtype=xp.int64)
    out.is_leaf = True
    out.grad_fn = None  # non-differentiable

    for hook in getattr(probs, "_activation_hooks", []):
        new_out = hook(out)
        if new_out is not None:
            out = new_out

    return out

def multinomial(probs: Tensor, num_samples: int = 1) -> Tensor:
    """
    Sample from categorical distribution.

    Args:
        probs (Tensor): Probability distribution of shape (B, V). Must be non-negative and sum to 1 along last dim.
        num_samples (int): Number of samples to draw per batch.

    Returns:
        Tensor: Sampled indices of shape (B, num_samples), dtype int64.
    """
    return dispatch_amp("multinomial", _multinomial_impl, probs, num_samples=num_samples)