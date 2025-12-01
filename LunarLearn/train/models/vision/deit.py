import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module
from LunarLearn.nn.transformer import VisionTransformer

xp = backend.xp

class DeiT(Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 n_classes=1000,
                 d_model=768,
                 n_heads=12,
                 n_layers=12,
                 ff_dim=None,
                 distillation=False,
                 pretrained=False,
                 **kwargs):
        super().__init__()
        ff_dim = ff_dim or (d_model * 4)
        self.vit = VisionTransformer(img_size=img_size,
                         patch_size=patch_size,
                         n_classes=n_classes,
                         d_model=d_model,
                         n_heads=n_heads,
                         n_layers=n_layers,
                         ff_dim=ff_dim,
                         use_output_head=True,
                         distillation=distillation,
                         **kwargs)

        if pretrained:
            self.load_state_dict(None)

    def forward(self, x):
        return self.vit(x)
    

class DeiTTiny(DeiT):
    def __init__(self, patch_size=16, n_classes=1000, distillation=False, pretrained=False):
        super().__init__(patch_size=patch_size,
                         n_classes=n_classes,
                         d_model=192,
                         n_heads=3,
                         distillation=distillation,
                         pretrained=pretrained)
        

class DeiTSmall(DeiT):
    def __init__(self, patch_size=16, n_classes=1000, distillation=False, pretrained=False):
        super().__init__(patch_size=patch_size,
                         n_classes=n_classes,
                         d_model=384,
                         n_heads=6,
                         distillation=distillation,
                         pretrained=pretrained)
        

class DeiTBase(DeiT):
    def __init__(self, patch_size=16, n_classes=1000, distillation=True, pretrained=False):
        super().__init__(patch_size=patch_size,
                         n_classes=n_classes,
                         d_model=768,
                         n_heads=12,
                         distillation=distillation,
                         pretrained=pretrained)
        

# Training loop
"""
teacher = RegNetY16GF(n_classes=1000)
teacher.eval()  # No gradients

# Student
student = DeiTBase(distillation=True)

optimizer = AdamW(student.parameters(), lr=5e-4)

for x, y in train_loader:
    student_logits_cls, student_logits_dist = student(x)

    with no_grad():
        teacher_logits = teacher(x)  # (B, 1000)

    # Soft labels
    teacher_probs = ops.softmax(teacher_logits / 4.0, axis=-1)  # T=4

    loss_cls = cross_entropy(student_logits_cls, y)
    loss_dist = soft_cross_entropy(student_logits_dist, teacher_probs)
    loss = 0.5 * loss_cls + 0.5 * loss_dist

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""