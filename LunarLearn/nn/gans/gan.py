from LunarLearn.core import ops
from LunarLearn.nn.gans import GANLoss, sample_noise, gradient_penalty
from LunarLearn.amp import amp
from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import Dense, BatchNorm, Activation, Conv2DTranspose, Conv2D, Flatten, LeakyReLU, Reshape

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
    def __init__(self, generator, discriminator, g_opt, d_opt, loss="vanilla", lambda_gp=10.0):
        self.G = generator
        self.D = discriminator
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.loss = GANLoss(mode=loss)
        self.lambda_gp = lambda_gp

    def step(self, real_batch, z_dim, d_steps=1, distribution="normal", truncation=1.0, enable_amp=True):
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
                if self.loss.mode == "wasserstein" and self.lambda_gp > 0:
                    gp = gradient_penalty(self.D, real_batch, fake, self.lambda_gp, enable_amp=enable_amp)
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

        return {"d_loss": d_loss.item(), "g_loss": g_loss.item()}

