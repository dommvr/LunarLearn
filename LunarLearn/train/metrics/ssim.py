import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _gaussian_filter, _gaussian_kernel
from LunarLearn.core import Tensor

xp = backend.xp


class SSIM(BaseMetric):
    def __init__(self,
                 data_range: float = 1.0,
                 kernel_size: int = 11,
                 sigma: float = 1.5,
                 eps: float = 1e-12):
        super().__init__(expect_vector=False)
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.eps = eps

    def compute(self, img1: Tensor, img2: Tensor):
        x = img1.data.astype(xp.float64)
        y = img2.data.astype(xp.float64)

        # Gaussian blur for local statistics
        mu_x = _gaussian_filter(x, self.kernel_size, self.sigma)
        mu_y = _gaussian_filter(y, self.kernel_size, self.sigma)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = _gaussian_filter(x * x, self.kernel_size, self.sigma) - mu_x2
        sigma_y2 = _gaussian_filter(y * y, self.kernel_size, self.sigma) - mu_y2
        sigma_xy = _gaussian_filter(x * y, self.kernel_size, self.sigma) - mu_xy

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + self.eps)

        # mean over channels and spatial dims
        return float(xp.mean(ssim_map))