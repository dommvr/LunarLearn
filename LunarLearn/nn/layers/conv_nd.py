import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = xp.DTYPE


class ConvND(BaseLayer):
    """
    Generic grouped convolution for 1D/2D/3D using im2col_nd + per-group GEMM.

    Input:  (m, C_in, *spatial)
    Output: (m, C_out, *out_spatial)

    Expects ops:
      - ops.pad(x, pad_width, mode='constant')
      - ops.im2col_nd(x, kernel_size, s, dilation=1, groups=1)  # you added dispatcher already
      - ops.matmul(A, B)
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
        if not (padding == 'same' or isinstance(padding, int)):
            raise ValueError("padding must be either 'same' or an integer")
        if isinstance(padding, int) and padding < 0:
            raise ValueError("padding must be >= 0")
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
        self.padding = padding
        self.dilation = _to_tuple(dilation, "dilation")
        self.bias = bias
        self.groups = groups

        self.activation = activation
        self.w_init = w_init
        self.uniform = uniform
        self.gain = gain

        super().__init__(trainable=True)

        self.W = None
        self.b = None

        self.pad = None           # tuple length dim
        self.out_spatial = None   # tuple length dim
        self.output_shape = None  # (C_out, *out_spatial)

    def initialize(self, input_shape):
        """
        input_shape: (C_in, *spatial)  (no batch)
        """
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

        # Store weights explicitly grouped to make forward trivial and correct:
        # (G, Cout_g, Cin_g, *kernel)
        W_shape = (self.groups, Cout_g, Cin_g, *self.kernel_size)
        b_shape = (self.filters, 1)

        W, b = initialize_weights(
            W_shape, b_shape,
            self.w_init, self.activation,
            self.uniform, self.gain
        )

        self.W = Parameter(W, requires_grad=True)
        self.b = Parameter(b, requires_grad=True) if self.bias else None
        self._apply_param_settings()

        # effective kernel sizes with dilation
        eff_k = [self.dilation[i] * (self.kernel_size[i] - 1) + 1 for i in range(self.dim)]

        # compute padding
        if self.padding == 'same':
            pad = []
            for i in range(self.dim):
                in_i = spatial[i]
                s_i = self.strides[i]
                eff_i = eff_k[i]
                p = ((s_i - 1) * in_i + eff_i - s_i) // 2
                pad.append(int(p))
            self.pad = tuple(pad)
        else:
            self.pad = (int(self.padding),) * self.dim

        # output spatial dims
        out_spatial = []
        for i in range(self.dim):
            in_i = spatial[i]
            p_i = self.pad[i]
            s_i = self.strides[i]
            eff_i = eff_k[i]
            out_i = (in_i + 2 * p_i - eff_i) // s_i + 1
            out_spatial.append(int(out_i))
        self.out_spatial = tuple(out_spatial)

        self.output_shape = (self.filters, *self.out_spatial)

    def forward(self, A_prev: Tensor) -> Tensor:
        from LunarLearn.nn.activations import get_activation

        if self.W is None:
            _, C_in, *spatial = A_prev.shape
            self.initialize((C_in, *spatial))

        W = self.W.to_compute()                          # (G, Cout_g, Cin_g, *k)
        b = self.b.to_compute() if self.b is not None else None

        m, C_in, *spatial = A_prev.shape
        G = self.groups
        Cin_g = C_in // G
        Cout_g = self.filters // G

        # pad_width: [(0,0),(0,0), (p,p), ...]
        pad_width = [(0, 0), (0, 0)]
        for p in self.pad:
            pad_width.append((p, p))
        A_pad = ops.pad(A_prev, tuple(pad_width), mode='constant')

        out_spatial = self.out_spatial
        outN = 1
        for v in out_spatial:
            outN *= v

        # output buffer
        # use backend dtype if possible
        out = xp.zeros((m, self.filters, *out_spatial), dtype=A_prev.data.dtype if hasattr(A_prev, "data") else DTYPE)

        # per-group GEMM
        for g in range(G):
            in_start = g * Cin_g
            in_end   = (g + 1) * Cin_g
            out_start = g * Cout_g
            out_end   = (g + 1) * Cout_g

            A_g = A_pad[:, in_start:in_end, *([slice(None)] * self.dim)]

            # Important: call im2col_nd with groups=1 for this slice.
            # This avoids the "stacked rows" mismatch problem.
            X_col = ops.im2col(A_g, self.kernel_size, self.strides, self.dilation, groups=1)
            # X_col: (Cin_g * prod(k), m * outN)

            W_g = W[g].reshape(Cout_g, -1)              # (Cout_g, Cin_g*prod(k))
            Z_col = ops.matmul(W_g, X_col)              # (Cout_g, m*outN)

            if b is not None:
                Z_col += b[out_start:out_end]           # (Cout_g, 1) broadcast

            # reshape columns back to (m, Cout_g, *out_spatial)
            # Current: (Cout_g, m*outN)
            Z = Z_col.reshape(Cout_g, *out_spatial, m)

            # move m to front: from (Cout_g, *out_spatial, m) -> (m, Cout_g, *out_spatial)
            perm = (self.dim + 1, 0, *range(1, self.dim + 1))
            Z = Z.transpose(perm)

            out[:, out_start:out_end, *([slice(None)] * self.dim)] = Z

        activation = get_activation(self.activation)
        return activation(out)


class Conv1D(ConvND):
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, dilation=1, bias=True):
        super().__init__(1, filters, kernel_size, strides, padding, activation,
                         w_init, uniform, gain, groups, dilation, bias)


class Conv2D(ConvND):
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, dilation=1, bias=True):
        super().__init__(2, filters, kernel_size, strides, padding, activation,
                         w_init, uniform, gain, groups, dilation, bias)


class Conv3D(ConvND):
    def __init__(self, filters, kernel_size, strides=1, padding=0, activation='linear',
                 w_init='auto', uniform=False, gain=1, groups=1, dilation=1, bias=True):
        super().__init__(3, filters, kernel_size, strides, padding, activation,
                         w_init, uniform, gain, groups, dilation, bias)