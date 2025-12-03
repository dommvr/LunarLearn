from LunarLearn.train.models.vision import VIT_B16
from LunarLearn.train.models.nlp import BERT
from LunarLearn.train.models.hybrid import CLIPImageEncoder, CLIPTextEncoder, CLIP, CLIPLoss

class VIT_B16_Encoder(VIT_B16):
    def __init__(self):
        super().__init__(use_output_head=False)

    def forward(self, x):
        cls_out, _ = super().forward(x)
        return cls_out
    

class BERTForCLIP(BERT):
    def __init__(self, pad_idx=None):
        super().__init__(use_mlm_head=False, use_nsp_head=False)
        self.pad_idx = pad_idx

    def forward(self, input_ids, attn_mask=None):
        hidden = super().forward(input_ids, self.pad_idx, return_hidden=True, return_attn=False)
        cls_emb = hidden[:, 0]
        return cls_emb


vision_backbone = VIT_B16_Encoder()

image_encoder = CLIPImageEncoder(vision_backbone,
                                 emb_dim=512,
                                 normalize=True)

nlp_backbone = BERTForCLIP()

text_encoder = CLIPTextEncoder(nlp_backbone,
                               emb_dim=512,
                               normalize=True)

clip_model = CLIP(image_encoder, text_encoder, temp=0.07)
clip_loss = CLIPLoss()

def train_step(batch):
    images = batch["images"]           # (B, C, H, W)
    input_ids = batch["input_ids"]     # (B, L)
    attention_mask = batch["attention_mask"]  # (B, L)

    optimizer.zero_grad()

    _, _, logits_per_image, logits_per_text = clip_model(
        images,
        input_ids,
        attention_mask
    )

    loss = clip_loss(logits_per_image, logits_per_text)
    loss.backward()
    optimizer.step()

    return float(loss)