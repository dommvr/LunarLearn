import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = xp.DTYPE


class ConvTransposeND(BaseLayer):
    """
    Generic transposed convolution (a.k.a. deconvolution) for 1D/2D/3D.

    Input:  (m, C_in, *spatial)
    Output: (m, C_out, *out_spatial)

    Uses im2col_transpose_nd / col2im_transpose_nd (or per-dim versions via dispatch).

    NOTE: This matches your current approach (im2col_transpose + matmul + col2im_transpose),
    but makes it consistent with ConvND conventions:
      - tuple-normalized kernel/stride/padding
      - grouped weights stored as (G, Cin_g, Cout_g, *k)
      - per-group computation
    """
    def __init__(self, dim, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, dilation=1, bias=True):
        from LunarLearn.nn.activations import activations
        from LunarLearn.nn.initializations import initializations

        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3")
        if not isinstance(filters, int) or filters <= 0:
            raise ValueError("filters must be a positive integer")
        if not isinstance(groups, int) or groups < 1:
            raise ValueError("groups must be >= 1")
        if not (padding == 'same' or isinstance(padding, int) or isinstance(padding, (tuple, list))):
            raise ValueError("padding must be 'same', int, or tuple/list")
        if activation not in activations.ACTIVATIONS:
            raise ValueError(f"Unsupported activation '{activation}'")
        if w_init not in initializations.ALL_INITIALIZATIONS:
            raise ValueError(f"Unsupported weight initialization '{w_init}'")

        def _to_tuple(x, name):
            if isinstance(x, int):
                return (x,) * dim
            if isinstance(x, (tuple, list)) and len(x) == dim:
                return tuple(int(v) for v in x)
            raise ValueError(f"{name} must be int or tuple/list of length {dim}")

        self.dim = dim
        self.filters = filters
        self.kernel_size = _to_tuple(kernel_size, "kernel_size")
        self.strides = _to_tuple(strides, "strides")
        self.padding = padding  # handled in initialize
        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain
        self.groups = groups
        self.dilation = dilation
        self.bias = bias

        super().__init__(trainable=True)

        self.W = None
        self.b = None

        self.pad = None           # tuple length dim
        self.out_spatial = None   # tuple length dim
        self.output_shape = None  # (C_out, *out_spatial)

    def initialize(self, input_shape):
        from LunarLearn.nn.initializations import initialize_weights

        if input_shape is None:
            raise ValueError("input_shape must be provided")

        C_in, *spatial = input_shape
        if len(spatial) != self.dim:
            raise ValueError(f"Expected {self.dim} spatial dims, got {len(spatial)}")

        if C_in % self.groups != 0:
            raise ValueError(f"Input channels ({C_in}) must be divisible by groups ({self.groups})")
        if self.filters % self.groups != 0:
            raise ValueError(f"Output filters ({self.filters}) must be divisible by groups ({self.groups})")

        Cin_g = C_in // self.groups
        Cout_g = self.filters // self.groups

        # For transposed conv, your original weight shape was (C_in, C_out, kH, kW).
        # Make it grouped: (G, Cin_g, Cout_g, *k)
        W_shape = (self.groups, Cin_g, Cout_g, *self.kernel_size)
        b_shape = (self.filters, 1)

        W, b = initialize_weights(
            W_shape, b_shape,
            self.w_init, self.activation,
            self.uniform, self.gain
        )

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True) if self.bias else None
        self._apply_param_settings()

        # Padding: keep behavior similar to your old code:
        # if padding == 'same': pad = floor(k/2) per dim
        if self.padding == 'same':
            self.pad = tuple(k // 2 for k in self.kernel_size)
        elif isinstance(self.padding, int):
            self.pad = (self.padding,) * self.dim
        else:
            # tuple/list padding
            if len(self.padding) != self.dim:
                raise ValueError(f"padding tuple must be length {self.dim}")
            self.pad = tuple(int(p) for p in self.padding)

        # Output spatial size for transposed conv (standard formula):
        # out = (in - 1) * stride - 2*pad + kernel
        out_spatial = []
        for i in range(self.dim):
            out_i = (spatial[i] - 1) * self.strides[i] - 2 * self.pad[i] + self.kernel_size[i]
            out_spatial.append(int(out_i))
        self.out_spatial = tuple(out_spatial)

        self.output_shape = (self.filters, *self.out_spatial)

    def forward(self, A_prev: Tensor) -> Tensor:
        from LunarLearn.nn.activations import get_activation

        if self.W is None:
            _, C_in, *spatial = A_prev.shape
            self.initialize((C_in, *spatial))

        W = self.W.to_compute()                          # (G, Cin_g, Cout_g, *k)
        b = self.b.to_compute() if self.b is not None else None

        m, C_in, *spatial = A_prev.shape
        G = self.groups
        Cin_g = C_in // G
        Cout_g = self.filters // G

        out_spatial = self.out_spatial
        out_shape_full = (m, self.filters, *out_spatial)

        out = xp.zeros(out_shape_full, dtype=A_prev.data.dtype if hasattr(A_prev, "data") else DTYPE)

        # per-group computation
        for g in range(G):
            in_start = g * Cin_g
            in_end   = (g + 1) * Cin_g
            out_start = g * Cout_g
            out_end   = (g + 1) * Cout_g

            A_g = A_prev[:, in_start:in_end, *([slice(None)] * self.dim)]    # (m, Cin_g, ...)

            X_col = ops.im2col_transpose(
                A_g, self.kernel_size, self.strides,
                output_shape=(m, Cout_g, *out_spatial),  # per-group output shape
                padding=self.pad, dilation=self.dilation
            )
            # For group: W[g] is (Cin_g, Cout_g, *k)
            W_g = W[g]  # (Cin_g, Cout_g, *k)
            W_col = W_g.reshape(Cin_g, -1).T            # (Cout_g*prod(k), Cin_g)

            out_col = ops.matmul(W_col, X_col)

            z_g = ops.col2im_transpose(
                out_col,
                input_shape=A_g.shape,                  # (m, Cin_g, ...)
                kernel_size=self.kernel_size,
                strides=self.strides,
                output_shape=(m, Cout_g, *out_spatial),
                padding=self.pad,
                dilation=self.dilation
            )  # (m, Cout_g, *out_spatial)

            out[:, out_start:out_end, *([slice(None)] * self.dim)] = z_g

        if b is not None:
            # b is (C_out, 1) -> broadcast to (m, C_out, *out_spatial)
            out += b.reshape(1, self.filters, *([1] * self.dim))

        activation = get_activation(self.activation)
        return activation(out)


class Conv1DTranspose(ConvTransposeND):
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, bias=True):
        super().__init__(1, filters, kernel_size, strides, padding, activation,
                         w_init, uniform, gain, groups, bias)


class Conv2DTranspose(ConvTransposeND):
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, bias=True):
        super().__init__(2, filters, kernel_size, strides, padding, activation,
                         w_init, uniform, gain, groups, bias)


class Conv3DTranspose(ConvTransposeND):
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, bias=True):
        super().__init__(3, filters, kernel_size, strides, padding, activation,
                         w_init, uniform, gain, groups, bias)