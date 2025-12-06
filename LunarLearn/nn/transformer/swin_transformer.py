from LunarLearn.nn import ModuleList
from LunarLearn.nn.layers import BaseLayer, ConvPatchEmbedding, Dropout, PatchMerging, LayerNorm, GlobalAveragePool2D, Dense
from LunarLearn.nn.transformer import SwinBlock
from LunarLearn.core import Tensor


class SwinTransformer(BaseLayer):
    def __init__(self,
                 patch_size=4,
                 n_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.0,
                 keep_prob=1.0,
                 norm_layer=LayerNorm,
                 use_head=True):
        super().__init__(trainable=True)

        self.n_classes = n_classes
        self.num_layers = len(depths)
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = ConvPatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)
        self.dropout = Dropout(keep_prob)
        self.layers = None

        self.norm = norm_layer() if norm_layer is not None else None
        self.avgpool = GlobalAveragePool2D()

        self.head = None
        if use_head and n_classes > 0:
            self.head = Dense(n_classes)
        
    def initialize(self, input_shape):
        img_size = input_shape[2]
        self.patches_resolution = (img_size // self.patch_embed.patch_size, img_size // self.patch_embed.patch_size)
        self.layers = ModuleList()
        for i in range(self.num_layers):
            layer = ModuleList([SwinBlock(dim=int(self.embed_dim * 2 ** i),
                                          num_heads=self.num_heads[i],
                                          window_size=self.window_size,
                                          shift_size=0 if (i % 2 == 0) else self.window_size // 2)
                                          for _ in range(self.depths[i])])
            self.layers.append(layer)

            if i < self.num_layers - 1:
                self.layers.append(PatchMerging(img_size // (self.patch_embed.patch_size * 2 ** i), int(self.embed_dim * 2 ** i)))

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)

        if self.layers is None:
            self.initialize(x.shape)
        
        x = self.dropout(x)

        H, W = self.patches_resolution
        for layer in self.layers:
            if isinstance(layer, PatchMerging):
                x = layer(x, H, W)
                H, W = H // 2, W // 2
            else:
                x = layer(x, H, W)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.flatten(1)

        if self.head is not None:
            x = self.head(x)
        return x