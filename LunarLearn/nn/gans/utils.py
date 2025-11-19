import LunarLearn.core.backend.backend as backend
from LunarLearn.core import ops
from LunarLearn.amp import amp

DTYPE = backend.DTYPE

def vanilla(d_real, d_fake):
    real_loss = ops.binary_cross_entropy_with_logits(d_real, ops.ones_like(d_real)) if d_real is not None else 0
    fake_loss = ops.binary_cross_entropy_with_logits(d_fake, ops.zeros_like(d_fake))
    d_loss = real_loss + fake_loss
    g_loss = ops.binary_cross_entropy_with_logits(d_fake, ops.ones_like(d_fake))
    return d_loss, g_loss

def lsgan(d_real, d_fake):
    real_loss = ops.mean_squared_error(d_real, ops.ones_like(d_real)) if d_real is not None else 0
    fake_loss = ops.mean_squared_error(d_fake, ops.zeros_like(d_fake))
    d_loss = real_loss + fake_loss
    g_loss = ops.mean_squared_error(d_fake, ops.ones_like(d_fake))
    return d_loss, g_loss

def hinge(d_real, d_fake):
    real_loss = ops.mean(ops.relu(1.0 - d_real)) if d_real is not None else 0
    fake_loss = ops.mean(ops.relu(1.0 + d_fake))
    d_loss = real_loss + fake_loss
    g_loss = -ops.mean(d_fake)
    return d_loss, g_loss

def wasserstein(d_real, d_fake):
    d_loss = (ops.mean(d_fake) - ops.mean(d_real)) if d_real is not None else 0
    g_loss = ops.mean(-d_fake)
    return d_loss, g_loss

def sample_noise(
    batch_size: int,
    z_dim: int,
    truncation: float = 1.0,
    distribution: str = "normal",
    dtype=DTYPE
):
    if distribution not in {"normal", "uniform", "truncated_normal"}:
        raise ValueError("distribution must be 'normal', 'uniform', or 'truncated_normal'")

    if distribution == "normal":
        z = ops.random_normal((batch_size, z_dim), dtype=dtype)

    elif distribution == "uniform":
        z = ops.random_uniform((batch_size, z_dim), -1, 1, dtype=dtype)

    elif distribution == "truncated_normal":
        z = ops.random_normal((batch_size, z_dim), dtype=dtype)
        if truncation < 1.0:
            z = ops.clip(z, -truncation, truncation)
            z = z * truncation

    return z

def gradient_penalty(disc, real, fake, lambda_gp=10.0, enable_amp=True):
    batch = real.shape[0]

    with amp.autocast(enabled=enable_amp):
        eps = ops.random_uniform((batch, 1, 1, 1), low=0.0, high=1.0)
        eps = ops.expand(eps, real.shape)
        interp = eps * real + (1 - eps) * fake

        interp = interp.detach()
        interp.requires_grad = True

        scores = disc(interp)
        dummy_loss = scores.sum()
        dummy_loss = amp.scale_loss(dummy_loss)
        dummy_loss.backward()

        grads = interp.grad
        grads_flat = grads.reshape(batch, -1)
        grad_norm = ops.sqrt(ops.sum(grads_flat ** 2, axis=1) + 1e-12)
        gp = ops.mean((grad_norm - 1.0) ** 2)
        return gp * lambda_gp

    