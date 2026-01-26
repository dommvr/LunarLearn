from LunarLearn.nn.layers import BaseLayer, Conv2D, LayerNorm
from LunarLearn.core import Tensor


class ConvPatchEmbedding(BaseLayer):
    def __init__(self, patch_size: int = 4, emb_dim: int = 96, norm: bool = True):
        super().__init__(trainable=True)
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.proj = Conv2D(emb_dim, kernel_size=patch_size, strides=patch_size, bias=False)
        self.norm = LayerNorm() if norm else None

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, emb_dim, H/p, W/p)
        out = x.flatten(2).transpose(0, 2, 1)  # (B, num_patches, emb_dim)
        if self.norm is not None:
            out = self.norm(out)
        return out