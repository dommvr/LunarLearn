from LunarLearn.nn.layers import BaseLayer
from LunarLearn.core import ops


class SpectralNorm(BaseLayer):
    def __init__(self, layer, n_power_iter=1):
        super().__init__(trainable=True)
        self.layer = layer
        self.n_power_iter = n_power_iter
        self.u = None

    def initialize(self, weight):
        h = weight.master.shape[0]
        u = ops.random_normal((1, h))
        self.u = u / ops.sqrt(ops.sum(u**2) + 1e-12)

    def forward(self, x):
        _ = self.layer(x)

        W = self.layer.W
        if W is None or W.master is None:
            return self.layer(x)

        weight = W.master

        if self.u is None:
            u = ops.random_normal((1, weight.shape[0]))
            self.u = (u / ops.sqrt(ops.sum(u**2) + 1e-12)).detach()

        u = self.u
        for _ in range(self.n_power_iter):
            v = ops.l2_normalize(weight.T @ u.T)
            u = ops.l2_normalize(weight @ v)
        sigma = ops.sum(u * (weight @ v)).item()   # scalar
        self.u = u.detach()

        # Apply to current compute tensor (fp16 or fp32)
        orig_compute = W.compute
        W.compute = W.compute / sigma
        try:
            return self.layer(x)
        finally:
            W.compute = orig_compute
