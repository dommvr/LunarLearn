import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module
from LunarLearn.nn.layers import Dense
from LunarLearn.nn.loss import BaseLoss
from LunarLearn.core import Tensor, Parameter, ops

xp = backend.xp
DTYPE = backend.DTYPE


class CLIPImageEncoder(Module):
    def __init__(self,
                 backbone: Module,
                 emb_dim: int,
                 normalize: bool = True):
        super().__init__()
        self.backbone = backbone
        self.proj = Dense(emb_dim)
        self.normalize = normalize

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        z = self.proj(feat)

        if self.normalize:
            z = ops.l2_normalize(z, axis=1)

        return z


class CLIPTextEncoder(Module):
    def __init__(self,
                 backbone: Module,
                 emb_dim: int,
                 normalize: bool = True):
        super().__init__()
        self.backbone = backbone
        self.proj = Dense(emb_dim)
        self.normalize = normalize

    def forward(self, input_ids: Tensor, attn_mask: Tensor = None) -> Tensor:
        feat = self.backbone(input_ids, attn_mask)
        z = self.proj(feat)

        if self.normalize:
            z = ops.l2_normalize(z, axis=1)

        return z


class CLIP(Module):
    def __init__(self,
                 image_encoder: Module,
                 text_encoder: Module,
                 temp: float = 0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        scale = xp.array(xp.log(1/temp), dtype=DTYPE)
        self.logit_scale = Parameter(scale, requires_grad=True)

    def encode_image(self, images: Tensor) -> Tensor:
        return self.image_encoder(images)

    def encode_text(self,
                    input_ids: Tensor,
                    attn_mask: Tensor = None) -> Tensor:
        return self.text_encoder(input_ids, attn_mask)

    def forward(self,
                images: Tensor,
                input_ids: Tensor,
                attn_mask: Tensor = None):
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attn_mask)

        logit_scale = ops.exp(self.logit_scale.to_compute())
        logit_per_image = logit_scale * ops.matmul(img_emb, txt_emb.T)
        logits_per_text = logit_per_image.T

        return img_emb, txt_emb, logit_per_image, logits_per_text
 

class CLIPLoss(BaseLoss):
    def __init__(self):
        from LunarLearn.nn.loss import CrossEntropy
        super().__init__(trainable=False)
        self.ce = CrossEntropy()

    def forward(self,
                logits_per_image: Tensor,
                logits_per_text: Tensor) -> Tensor:
        B = logits_per_image.shape[0]
        labels = xp.arange(B, dtype=xp.int64)

        loss_i = self.ce(logits_per_image, labels)
        loss_t = self.ce(logits_per_text, labels)
        loss = 0.5 * (loss_i + loss_t)
        return loss
