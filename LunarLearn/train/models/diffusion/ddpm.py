import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module
from LunarLearn.nn.layers import BaseLayer, BatchNorm2D
from LunarLearn.train.models.segmentation import UNet
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class DiffusionSchedule:
    """
    Precomputes diffusion coefficients for a simple linear beta schedule.
    Timesteps: 0..T-1
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2):
        self.T = T

        self.betas = xp.linspace(beta_start, beta_end, T, dtype=DTYPE)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = xp.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = xp.concatenate(
            [xp.array([1.0], dtype=DTYPE), self.alphas_cumprod[:-1]],
            axis=0
        )

        self.sqrt_alphas_cumprod = xp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = xp.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_inv_alphas = xp.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _gather(self, arr, t):
        # arr: (T,), t: (B,)
        return arr[t.astype(xp.int64)]

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
        """
        sqrt_ab = self._gather(self.sqrt_alphas_cumprod, t)
        sqrt_omb = self._gather(self.sqrt_one_minus_alphas_cumprod, t)

        while sqrt_ab.ndim < x0.ndim:
            sqrt_ab = sqrt_ab[..., None]
            sqrt_omb = sqrt_omb[..., None]

        return sqrt_ab * x0 + sqrt_omb * noise

    def p_mean_variance(self, model, x_t: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Mean/variance of p_theta(x_{t-1} | x_t) using epsilon-predicting model.
        """
        eps_theta = model(x_t, t)  # predict noise

        betas_t = self._gather(self.betas, t)
        sqrt_inv_alpha_t = self._gather(self.sqrt_inv_alphas, t)
        sqrt_one_minus_ab_t = self._gather(self.sqrt_one_minus_alphas_cumprod, t)
        alpha_bar_prev_t = self._gather(self.alphas_cumprod_prev, t)
        alpha_bar_t = self._gather(self.alphas_cumprod, t)

        for arr_name in ["betas_t", "sqrt_inv_alpha_t", "sqrt_one_minus_ab_t",
                         "alpha_bar_prev_t", "alpha_bar_t"]:
            arr = locals()[arr_name]
            while arr.ndim < x_t.ndim:
                arr = arr[..., None]
            locals()[arr_name] = arr

        betas_t = locals()["betas_t"]
        sqrt_inv_alpha_t = locals()["sqrt_inv_alpha_t"]
        sqrt_one_minus_ab_t = locals()["sqrt_one_minus_ab_t"]
        alpha_bar_prev_t = locals()["alpha_bar_prev_t"]
        alpha_bar_t = locals()["alpha_bar_t"]

        coef1 = sqrt_inv_alpha_t * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
        coef2 = sqrt_inv_alpha_t * betas_t / (1.0 - alpha_bar_t)

        mean = coef1 * x_t - coef2 * eps_theta

        var = self._gather(self.posterior_variance, t)
        while var.ndim < x_t.ndim:
            var = var[..., None]

        return mean, var

    def p_sample(self, model, x_t: Tensor, t: Tensor) -> Tensor:
        """
        One reverse step: sample x_{t-1} from p_theta(x_{t-1} | x_t).
        """
        mean, var = self.p_mean_variance(model, x_t, t)
        if (t == 0).all():
            return mean
        noise = Tensor(xp.random.randn(*x_t.shape), requires_grad=False, dtype=x_t.dtype)
        return mean + xp.sqrt(var) * noise
    

class TimeChannel(BaseLayer):
    """
    Converts scalar timesteps t (B,) into a single extra channel, normalized and
    broadcast to spatial size of x.
    """
    def __init__(self, T: int):
        super().__init__(trainable=False)
        self.T = T

    def initialize(self, input_shape):
        # input_shape is not really used; we depend on x at forward
        self.output_shape = None

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        x: (B, C, H, W)
        t: (B,) (int or float)
        returns: (B, C+1, H, W)
        """
        B, _, H, W = x.shape
        t_norm = (t.astype(DTYPE) / (self.T - 1)).reshape(B, 1, 1, 1)
        t_map = xp.ones((B, 1, H, W), dtype=DTYPE) * t_norm
        t_map = Tensor(t_map, requires_grad=False, dtype=x.dtype)
        return ops.concatenate([x, t_map], axis=1)
    

class UNetEpsPredictor(Module):
    """
    Wraps your UNet to be used as an epsilon-predictor in DDPM.
    Adds a time channel to the input and predicts noise with same shape as x.
    """
    def __init__(self,
                 in_channels=3,
                 filters=64,
                 depth=4,
                 mode="bilinear",
                 norm_layer=BatchNorm2D,
                 activation="relu",
                 T=1000):
        super().__init__()

        # time channel mapper
        self.time_channel = TimeChannel(T=T)

        # UNet: num_classes = in_channels, final_activation = linear
        # so output shape = (B, in_channels, H, W), no squash
        self.unet = UNet(
            num_classes=in_channels,
            filters=filters,
            depth=depth,
            mode=mode,
            norm_layer=norm_layer,
            activation=activation,
            final_activation="linear",  # important: no sigmoid/softmax
            pretrained=False
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # concat time channel
        x_tc = self.time_channel(x, t)
        eps = self.unet(x_tc)
        return eps
    

class DDPM(Module):
    """
    Full DDPM model:
    - `eps_model`: epsilon-predictor (e.g. UNetEpsPredictor)
    - `schedule`: DiffusionSchedule (betas/alphas, q/p)
    """
    def __init__(self,
                 eps_model: Module,
                 schedule: DiffusionSchedule):
        super().__init__()
        self.eps_model = eps_model
        self.schedule = schedule

    def forward(self, x0: Tensor) -> Tensor:
        """
        Training forward:
        - sample t ~ Uniform{0..T-1}
        - sample noise ~ N(0, I)
        - compute x_t
        - predict noise
        - return MSE loss between true noise and predicted noise
        """
        B = x0.shape[0]
        T = self.schedule.T

        t_np = xp.random.randint(0, T, size=(B,), dtype=xp.int64)
        t = Tensor(t_np, requires_grad=False, dtype=DTYPE)

        noise_np = xp.random.randn(*x0.shape)
        noise = Tensor(noise_np, requires_grad=False, dtype=x0.dtype)

        x_t = self.schedule.q_sample(x0, t, noise)
        eps_pred = self.eps_model(x_t, t)

        diff = eps_pred - noise
        loss = ops.mean(diff * diff)
        return loss

    def sample(self, shape) -> Tensor:
        """
        Ancestral sampling: x_T ~ N(0, I), then iterate to x_0.
        shape: (B, C, H, W)
        """
        B = shape[0]
        T = self.schedule.T
        x_np = xp.random.randn(*shape)
        x_t = Tensor(x_np, requires_grad=False, dtype=DTYPE)

        for t_step in reversed(range(T)):
            t_np = xp.full((B,), t_step, dtype=xp.int64)
            t = Tensor(t_np, requires_grad=False, dtype=DTYPE)
            x_t = self.schedule.p_sample(self.eps_model, x_t, t)

        return x_t