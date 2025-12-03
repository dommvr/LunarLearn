from LunarLearn.nn import Module
from LunarLearn.nn.transformer import VisionTransformer
from LunarLearn.nn.layers import LayerNorm
from LunarLearn.nn.transformer.attention import ScaledDotProductAttention
from LunarLearn.core import Tensor


class VIT_B16(Module):
    def __init__(self,
                 img_size=224,
                 n_classes=1000,
                 keep_prob=1.0,
                 pos_mode="learnable",
                 norm=LayerNorm,
                 norm_position="post",
                 enc_share_weights=False,
                 use_output_head=True,
                 distillation=False,
                 ff_activation="gelu",
                 **kwargs):
        super().__init__()
        self.vit = VisionTransformer(img_size=img_size,
                                     patch_size=16,
                                     n_classes=n_classes,
                                     d_model=768,
                                     n_heads=12,
                                     pos_mode=pos_mode,
                                     n_layers=12,
                                     ff_dim=3072,
                                     ff_activation=ff_activation,
                                     keep_prob=keep_prob,
                                     attention=ScaledDotProductAttention,
                                     norm=norm,
                                     norm_position=norm_position,
                                     enc_share_weights=enc_share_weights,
                                     use_output_head=use_output_head,
                                     res_scale=1.0,
                                     distillation=distillation,
                                     **kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.vit(x)