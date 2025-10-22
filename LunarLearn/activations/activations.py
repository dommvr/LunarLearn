import LunarLearn.backend as backend
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import ops

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
MIXED_PRECISION = backend.MIXED_PRECISION

# -----------------------------
# Basic activations
# -----------------------------
def linear(x: Tensor) -> Tensor:
    """
    Linear activation (identity).

    Returns the input tensor unchanged.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Same as input tensor.

    Notes:
        - Autograd tracks this operation automatically.
        - Sets `Tensor.grad_fn = 'linear'` for computation graph traceability.
    """
    out = Tensor(x.data, requires_grad=x.requires_grad, dtype=x.dtype)
    out.grad_fn = "linear"
    out._prev = {x}
    return out

def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation.

    Computes the elementwise sigmoid function: 1 / (1 + exp(-x)).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with values in (0, 1).

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'sigmoid'`.
    """
    out = 1 / (1 + ops.exp(-x))
    out.grad_fn = "sigmoid"
    return out

def relu(x: Tensor) -> Tensor:
    """
    ReLU activation.

    Computes elementwise max(0, x).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Non-negative tensor with same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'relu'`.
    """
    out = ops.maximum(0, x)
    out.grad_fn = "relu"
    return out

def leaky_relu(x: Tensor, alpha=0.01) -> Tensor:
    """
    Leaky ReLU activation.

    Computes elementwise x if x > 0 else alpha * x.

    Args:
        x (Tensor): Input tensor.
        alpha (float, optional): Slope for negative values. Default is 0.01.

    Returns:
        Tensor: Tensor with same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'leaky_relu'`.
    """
    alpha = xp.array(alpha, dtype=x.dtype)
    out = ops.where(x > 0, x, x * alpha)
    out.grad_fn = "leaky_relu"
    return out

def tanh(x: Tensor) -> Tensor:
    """
    Tanh activation.

    Computes the elementwise hyperbolic tangent: (exp(x) - exp(-x)) / (exp(x) + exp(-x)).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor with values in (-1, 1).

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'tanh'`.
    """
    out = ops.tanh(x)
    return out

# -----------------------------
# Advanced activations
# -----------------------------
def softmax(x: Tensor, axis=-1) -> Tensor:
    """
    Softmax activation.

    Computes the softmax of the input tensor along the given axis.

    Args:
        x (Tensor): Input tensor.
        axis (int, optional): Axis along which to normalize. Default is -1.

    Returns:
        Tensor: Tensor with same shape as input and values summing to 1 along the specified axis.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'softmax'`.
    """
    exp_shifted = ops.exp(x - ops.max(x, axis=axis, keepdims=True))
    out = exp_shifted / ops.sum(exp_shifted, axis=axis, keepdims=True)
    out.grad_fn = "softmax"
    return out

def log_softmax(x: Tensor, axis=-1) -> Tensor:
    """
    Log-Softmax activation.

    Computes log of softmax along the given axis for numerical stability.

    Args:
        x (Tensor): Input tensor.
        axis (int, optional): Axis along which to normalize. Default is -1.

    Returns:
        Tensor: Tensor of same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'log_softmax'`.
    """
    m = ops.max(x, axis=axis, keepdims=True)
    shifted = x - m
    logsumexp = ops.log(ops.sum(ops.exp(shifted), axis=axis, keepdims=True))
    out = shifted - logsumexp
    out.grad_fn = "log_softmax"
    return out

def swish(x: Tensor) -> Tensor:
    """
    Swish activation.

    Computes x * sigmoid(x).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor of same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'swish'`.
    """
    sig = 1 / (1 + ops.exp(-x))
    out = x * sig
    out.grad_fn = "swish"
    return out

def mish(x: Tensor) -> Tensor:
    """
    Mish activation.

    Computes x * tanh(softplus(x)).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor of same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'mish'`.
    """
    sp = ops.log(1 + ops.exp(x))
    out = x * ops.tanh(sp)
    out.grad_fn = "mish"
    return out

def gelu(x: Tensor, approximate: bool = True) -> Tensor:
    """
    GELU activation.

    Computes the Gaussian Error Linear Unit.

    Args:
        x (Tensor): Input tensor.
        approximate (bool, optional): Use approximate formula. Default is True.

    Returns:
        Tensor: Tensor of same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'gelu'`.
    """
    if approximate:
        coeff = (2 / xp.pi) ** 0.5
        out = 0.5 * x * (1 + ops.tanh(coeff * (x + 0.044715 * (x ** 3))))
    else:
        out = 0.5 * x * (1 + ops.erf(x / (2 ** 0.5)))
    out.grad_fn = "gelu"
    return out

def softplus(x: Tensor) -> Tensor:
    """
    Softplus activation.

    Computes log(1 + exp(x)) elementwise.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor of same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'softplus'`.
    """
    out = ops.log(1 + ops.exp(x))
    out.grad_fn = "softplus"
    return out

def elu(x: Tensor, alpha=1.0) -> Tensor:
    """
    ELU activation.

    Computes x if x > 0 else alpha * (exp(x) - 1).

    Args:
        x (Tensor): Input tensor.
        alpha (float, optional): ELU parameter. Default is 1.0.

    Returns:
        Tensor: Tensor of same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'elu'`.
    """
    out = ops.where(x > 0, x, alpha * (ops.exp(x) - 1))
    out.grad_fn = "elu"
    return out

def selu(x: Tensor) -> Tensor:
    """
    SELU activation.

    Computes scaled ELU: scale * (x if x>0 else alpha*(exp(x)-1)).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Tensor of same shape as input.

    Notes:
        - Fully compatible with autograd.
        - Sets `Tensor.grad_fn = 'selu'`.
    """
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    out = scale * ops.where(x > 0, x, alpha * (ops.exp(x) - 1))
    out.grad_fn = "selu"
    return out

ACTIVATIONS = {
    "None": linear,
    "linear": linear,
    "sigmoid": sigmoid,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "softmax": softmax,
    "log_softmax": log_softmax,
    "swish": swish,
    "mish": mish,
    "gelu": gelu,
    "softplus": softplus,
    "elu": elu,
    "selu": selu,
}

def get_activation(name_or_fn):
    """
    Fetch an activation function by name or return it directly if already callable.

    Args:
        name_or_fn (str or callable): 
            - If str, returns the corresponding activation function.
            - If callable, returns it directly (assumes correct signature).

    Returns:
        callable: The activation function.
    """
    if callable(name_or_fn):
        return name_or_fn
    if isinstance(name_or_fn, str):
        if name_or_fn not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{name_or_fn}'. "
                             f"Available: {list(ACTIVATIONS.keys())}")
        return ACTIVATIONS[name_or_fn]
    raise TypeError("Activation must be a string or a callable")