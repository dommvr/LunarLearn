from LunarLearn.nn import Module
from LunarLearn.nn.layers import LayerNorm
from LunarLearn.nn.transformer import SwinTransformer
from LunarLearn.core import Tensor


class Swin(Module):
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
                 use_head=True,
                 pretrained=False):
        super().__init__()
        self.swin = SwinTransformer(patch_size,
                                    n_classes,
                                    embed_dim,
                                    depths,
                                    num_heads,
                                    window_size,
                                    mlp_ratio,
                                    keep_prob,
                                    norm_layer,
                                    use_head)
        
        if pretrained:
            self.load_state_dict(None)

    def forward(self, x: Tensor) -> Tensor:
        return self.swin(x)
    

class SwinTiny(Swin):
    def __init__(self, patch_size, pretrained=False, **kwargs):
        super().__init__(patch_size=patch_size,
                         embed_dim=96,
                         depths=[2, 2, 6, 2],
                         num_heads=[3, 6, 12, 24],
                         window_size=7,
                         pretrained=pretrained,
                         **kwargs)
        

class SwinSmall(Swin):
    def __init__(self, patch_size, pretrained=False, **kwargs):
        super().__init__(patch_size=patch_size,
                         embed_dim=96,
                         depths=[2, 2, 18, 2],
                         num_heads=[3, 6, 12, 24],
                         window_size=7,
                         pretrained=pretrained,
                         **kwargs)
        

class SwinBase(Swin):
    def __init__(self, patch_size, pretrained=False, **kwargs):
        super().__init__(patch_size=patch_size,
                         embed_dim=128,
                         depths=[2, 2, 18, 2],
                         num_heads=[4, 8, 16, 32],
                         window_size=7,
                         pretrained=pretrained,
                         **kwargs)