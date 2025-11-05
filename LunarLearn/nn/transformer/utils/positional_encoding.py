import LunarLearn.backend as backend

xp = backend.xp

def apply_rope(Q, K):
    seq_len = Q.shape[-2]
    dim = Q.shape[-1]
    pos = xp.arange(seq_len)[:, None]
    freqs = 1.0 / (10000 ** (xp.arange(0, dim, 2) / dim))
    angles = pos * freqs
    cos = xp.cos(angles)[None, None, :, :]
    sin = xp.sin(angles)[None, None, :, :]

    def rotate(x):
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return xp.stack([-x_odd, x_even], axis=-1).reshape(x.shape)

    Q_rot = Q * cos + rotate(Q) * sin
    K_rot = K * cos + rotate(K) * sin
    return Q_rot, K_rot

def get_alibi_bias(Q, K):
    n_heads, seq_len = Q.shape[1], Q.shape[2]
    slopes = xp.array([2 ** (-(2 ** -(i + 3))) for i in range(n_heads)], dtype=Q.dtype)
    distance = xp.arange(seq_len)[None, :] - xp.arange(seq_len)[:, None]
    bias = -xp.abs(distance)[None, None, :, :] * slopes[:, None, None, None]
    return bias