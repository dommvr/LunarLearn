import LunarLearn.core.backend.backend as backend
from LunarLearn.core import ops
from LunarLearn.nn.gans import GANLoss, sample_noise, gradient_penalty
from LunarLearn.amp import amp
from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import Dense, BatchNorm, Activation, Conv2DTranspose, Conv2D, Flatten, LeakyReLU, Reshape
import copy

xp = backend.xp

class Generator(Module):
    def __init__(self, channels=3, base=64):
        super().__init__()
        self.net = ModuleList([
            Dense(base*8*4*4), BatchNorm(), Activation("relu"),
            Reshape(-1, base*8, 4, 4),
            Conv2DTranspose(base*4, 4, strides=2, padding=1), BatchNorm(), Activation('relu'),
            Conv2DTranspose(base*2, 4, strides=2, padding=1), BatchNorm(), Activation('relu'),
            Conv2DTranspose(base, 4, strides=2, padding=1), BatchNorm(), Activation('relu'),
            Conv2DTranspose(channels, 4, strides=2, padding=1), Activation('tanh')
        ])

    def forward(self, z):
        return self.net(z)
    

class Discriminator(Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = ModuleList([
            Conv2D(base, 4, strides=2, padding=1), LeakyReLU(alpha=0.2),
            Conv2D(base, 4, strides=2, padding=1), LeakyReLU(alpha=0.2),
            Conv2D(base, 4, strides=2, padding=1), LeakyReLU(alpha=0.2),
            Conv2D(base, 4, strides=2, padding=1), LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(1)
        ])

    def forward(self, z):
        return self.net(z)


class GAN:
    def __init__(self, generator, discriminator, g_opt, d_opt, loss="vanilla", mode_gp="r1", gamma_gp=10.0, use_ema=True, ema_decay=0.999):
        self.G = generator
        self.G_ema = copy.deepcopy(generator) if use_ema else None
        self.D = discriminator
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.loss = GANLoss(mode=loss)
        self.mode_gp = mode_gp
        self.gamma_gp = gamma_gp
        self.ema_decay = ema_decay

    def _update_ema(self):
        for p, p_ema in zip(self.G.parameters(), self.G_ema.parameters()):
            p_ema.master.data = self.ema_decay * p_ema.master.data + (1 - self.ema_decay) * p.master.data

    def step(self, real_batch, z_dim, d_steps=1, distribution="normal", truncation=1.0, enable_amp=True):
        self.G.train()
        self.D.train()
        batch_size = real_batch.shape[0]

        # === Discriminator ===
        for _ in range(d_steps):
            with amp.autocast(enabled=enable_amp):
                self.D.zero_grad()

                d_real = self.D(real_batch)
                z = sample_noise(batch_size, z_dim, truncation=truncation, distribution=distribution)
                fake = self.G(z).detach()
                d_fake = self.D(fake)

                d_loss, _ = self.loss(d_real, d_fake)
                if self.loss.mode == "wasserstein" and self.gamma_gp > 0:
                    gp = gradient_penalty(self.D, real_batch, fake if self.mode_gp == "standard" else None,
                                          gamma=self.gamma_gp, mode=self.mode_gp, enable_amp=enable_amp)
                    d_loss = d_loss + gp
                d_loss = amp.scale_loss(d_loss)
                d_loss.backward()
                amp.step_if_ready(self.d_opt, self.D)
                
        # === Generator ===
        with amp.autocast(enabled=enable_amp):
            self.G.zero_grad()

            z = sample_noise(batch_size, z_dim, truncation=truncation, distribution=distribution)
            fake = self.G(z)
            d_fake = self.D(fake)

            _, g_loss = self.loss(None, d_fake)
            g_loss = amp.scale_loss(g_loss)
            g_loss.backward()
            amp.step_if_ready(self.g_opt, self.G)

        if self.G_ema is not None:
            self._update_ema()

        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}
    
    def generate(self, n_images=64, z_dim=128, truncation=1.0, use_ema=True, to_img=False, enable_amp=True):
        with backend.no_grad():
            with amp.autocast(enabled=enable_amp):
                z = sample_noise(n_images, z_dim, truncation, distribution="normal")
                if use_ema and self.G_ema is not None:
                    self.G_ema.eval()
                    fake = self.G_ema(z)
                else:
                    self.G.eval()
                    fake = self.G(z)
                
                if to_img:
                    fake = (ops.clip(fake, -1, 1) + 1) / 2
                    fake = (fake * 255).astype(xp.uint8)
                    fake = fake.transpose(0, 2, 3, 1)

                return fake
