from LunarLearn.nn.layers import BaseLayer, Dense
from LunarLearn.core import Tensor

class PatchEmbedding(BaseLayer):
    def __init__(self, patch_size: int = 16, emb_dim: int = 768):
        super().__init__(trainable=True)
        self.patch_size = patch_size
        self.grid_h = None
        self.grid_w = None
        self.num_patches = None
        self.emb_dim = emb_dim
        self.projection = Dense(emb_dim)

    def _initialize(self, input_shape):
        C, H, W = input_shape
        self.grid_h = H // self.patch_size
        self.grid_w = W // self.patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.output_shape = (self.num_patches, self.emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.num_patches is None:
            self._initialize(x.shape[1:])

        B, C, _, _ = x.shape
        x = x.reshape(B, C, self.grid_h, self.patch_size, self.grid_w, self.patch_size)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, self.num_patches, -1)
        out = self.projection(x)
        return out