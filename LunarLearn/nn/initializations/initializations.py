import LunarLearn.core.backend.backend as backend

xp = backend.xp
DTYPE = backend.DTYPE

def calculate_fan_in_and_fan_out(shape):
    """
    Compute fan_in and fan_out from a given weight shape.
    Works for Dense and Conv layers (1D, 2D, 3D).
    """
    if len(shape) == 2:  # Dense: (fan_in, fan_out)
        fan_in, fan_out = shape[0], shape[1]
    elif len(shape) >= 3:  
        # Conv: (kH, kW, ..., in_channels, out_channels)
        receptive_field_size = xp.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    else:
        # fallback: treat as scalar
        fan_in = fan_out = 1
    return fan_in, fan_out

def He(shape, uniform=False):
    fan_in, _ = calculate_fan_in_and_fan_out(shape)
    if uniform:
        limit = xp.sqrt(6.0 / fan_in, dtype=DTYPE)
        return xp.random.uniform(-limit, limit, shape).astype(DTYPE)
    else:
        return xp.random.randn(*shape).astype(DTYPE) * xp.sqrt(2.0 / fan_in, dtype=DTYPE)

def Xavier(shape, uniform=True):
    fan_in, fan_out = calculate_fan_in_and_fan_out(shape)
    if uniform:
        limit = xp.sqrt(6.0 / (fan_in + fan_out), dtype=DTYPE)
        return xp.random.uniform(-limit, limit, shape).astype(DTYPE)
    else:
        return xp.random.randn(*shape).astype(DTYPE) * xp.sqrt(2.0 / (fan_in + fan_out), dtype=DTYPE)

def LeCun(shape, uniform=False):
    fan_in, _ = calculate_fan_in_and_fan_out(shape)
    if uniform:
        limit = xp.sqrt(3.0 / fan_in, dtype=DTYPE)
        return xp.random.uniform(-limit, limit, shape).astype(DTYPE)
    else:
        return xp.random.randn(*shape).astype(DTYPE) * xp.sqrt(1.0 / fan_in, dtype=DTYPE)

def Orthogonal(shape, gain=1.0):
    """
    Orthogonal initialization.
    - gain=1.0 for tanh
    - gain=sqrt(2) for ReLU
    """
    if len(shape) < 2:
        raise ValueError("Orthogonal initializer requires at least 2D shape")

    flat_shape = (shape[0], int(xp.prod(shape[1:])))
    a = xp.random.randn(*flat_shape).astype(DTYPE)

    # QR decomposition
    q, r = xp.linalg.qr(a)
    q = q.astype(DTYPE)
    r = r.astype(DTYPE)

    # Make Q uniform (fix sign)
    q *= xp.sign(xp.diag(r))

    return (q.reshape(shape)) * gain

INITIALIZATION = {
    "he": He,
    "xavier": Xavier,
    "lecun": LeCun,
    "orthogonal": Orthogonal
}

ALIASES = {
    "he_normal": He,
    "he_uniform": lambda shape: He(shape, uniform=True),
    "xavier_normal": lambda shape: Xavier(shape, uniform=False),
    "xavier_uniform": Xavier,
    "lecun_normal": LeCun,
    "lecun_uniform": lambda shape: LeCun(shape, uniform=True),
}

ALL_INITIALIZATIONS = {**INITIALIZATION, **ALIASES}

AUTO_INIT_MAP = {
    "linear": "xavier",
    "sigmoid": "xavier",
    "relu": "he",
    "leaky_relu": "he",
    "tanh": "xavier",
    "softmax": "xavier",
    "log_softmax": "xavier",
    "swish": "he",
    "mish": "he",
    "gelu": "he",
    "softplus": "xavier",
    "elu": "he",
    "selu": "lecun",
}

def get_initialization(name_or_fn, activation: str = None):
    """
    Return an initialization function.
    - If `name_or_fn` is callable, return it directly.
    - If it's "auto", choose based on activation.
    - If it's a string, resolve it against initializers + aliases.
    """
    if callable(name_or_fn):
        return name_or_fn

    if isinstance(name_or_fn, str):
        name_or_fn = name_or_fn.lower()

        if name_or_fn == "auto":
            if activation is None:
                raise ValueError(
                    "'auto' initialization requires the layer's activation function "
                    "to be specified."
                )
            act = activation.lower()
            if act not in AUTO_INIT_MAP:
                raise ValueError(
                    f"No automatic initializer mapping found for activation '{activation}'. "
                    f"Supported activations: {list(AUTO_INIT_MAP.keys())}"
                )
            name_or_fn = AUTO_INIT_MAP[act]

        if name_or_fn not in ALL_INITIALIZATIONS:
            raise ValueError(
                f"Unsupported weight initialization '{name_or_fn}'. "
                f"Available: {list(INITIALIZATION.keys())}, "
                f"Aliases: {list(ALIASES.keys())}, "
                f"'auto' (with activation)"
            )
        return ALL_INITIALIZATIONS[name_or_fn]

    raise TypeError("Weight initialization must be a string or a callable")

def initialize_weights(w_shape, b_shape, w_init, activation, uniform=False, gain=1.0, zero_bias=False, bias_value=None, bias=True):
    """
    Initialize weights and biases based on chosen scheme and activation.

    Args:
        w_shape (tuple): Shape of the weights.
        b_shape (tuple): Shape of the biases.
        w_init (str or callable): Weight initializer name, function, or "auto".
        activation (str): Activation function name.
        uniform (bool): Whether to use uniform distribution where applicable.
        gain (float): Gain multiplier (useful for orthogonal init).

    Returns:
        W (ndarray): Initialized weights.
        b (ndarray): Initialized biases.
    """
    # Get initializer function
    init_fn = get_initialization(w_init)

    # Initialize weights
    W = init_fn(w_shape, uniform=uniform) if w_init != "orthogonal" else init_fn(w_shape, gain=gain)

    if bias:
        if bias_value is not None:
            b = xp.full(b_shape, bias_value, dtype=DTYPE)
        # Initialize biases depending on activation
        elif not zero_bias and activation in {"relu", "leaky_relu", "elu"}:
            # Small positive bias helps ReLU family avoid dead neurons
            b = xp.full(b_shape, 0.01, dtype=DTYPE)
        else:
            # Safe default: zeros
            b = xp.zeros(b_shape, dtype=DTYPE)
    else:
        b = None

    return W, b
