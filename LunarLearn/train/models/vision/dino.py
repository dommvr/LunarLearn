import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Module, ModuleList
from LunarLearn.nn.layers import BaseLayer, Dense, LayerNorm, Activation
from LunarLearn.nn.transformer import VisionTransformer
from LunarLearn.core import ops
import copy

xp = backend.xp

class DINOHead(BaseLayer):
    def __init__(self,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256,
                 out_dim: int = 65536,
                 norm_last: bool = True):
        super().__init__(trainable=True)

        self.head = ModuleList([Dense(hidden_dim, activation="gelu"),
                                Dense(hidden_dim, activation="gelu"),
                                Dense(bottleneck_dim)])
        
        self.norm = LayerNorm() if norm_last else None
        self.dense = Dense(out_dim, bias=False, weight_norm=True)

    def forward(self, x):
        x = self.head(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.dense(x)
        out = ops.l2_normalize(x, axis=1)
        return out
        

class DINO(Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 n_layers: int,
                 out_dim: int = 65536,
                 hidden_dim: int = 2048,
                 bottleneck_dim: int = 256,
                 momentum: float = 0.996,
                 center_momentum: float = 0.9,
                 temperature: float = 0.1,
                 use_multi_crop: bool = False,
                 pretrained: bool = False):
        super().__init__()

        self.backbone = VisionTransformer(d_model=d_model,
                                          n_heads=n_heads,
                                          n_layers=n_layers)
        self.student_head = DINOHead(hidden_dim, bottleneck_dim, out_dim)
        self.teacher_head = DINOHead(hidden_dim, bottleneck_dim, out_dim)

        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_backbone.eval()
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        self.momentum = momentum
        self.center_momentum = center_momentum
        self.temperature = temperature
        self.use_multi_crop = use_multi_crop

        self.center = ops.zeros((1, out_dim))

        if pretrained:
            self.load_state_dict(None)

    @backend.no_grad()
    def update_teacher(self):
        for s_param, t_param in zip(self.backbone.parameters(),
                                    self.teacher_backbone.parameters()):
            t_param.master = self.momentum * t_param.master + (1.0 - self.momentum) * s_param.master

        for s_param, t_param in zip(self.student_head.parameters(),
                                    self.teacher_head.parameters()):
            t_param.master = self.momentum * t_param.master + (1.0 - self.momentum) * s_param.master

    @backend.no_grad()
    def update_center(self, teacher_out):
        batch_center = ops.mean(teacher_out, axis=0, keepdims=True)
        self.center = self.center_momentum * self.center + (1 - self.center_momentum) * batch_center

    def forward(self, views):
        if not isinstance(views, (list, tuple)):
            views = [views]

        student_outs = []
        teacher_outs = []

        for x in views:
            s_feat = self.backbone(x)[:, 0]
            s_out = self.student_head(s_feat)
            s_out = s_out / self.temperature
            student_outs.append(s_out)

            with backend.no_grad():
                t_feat = self.teacher_backbone(x)[:, 0]
                t_out = self.teacher_head(t_feat)
                t_out = (t_out - self.center) / self.temperature
                teacher_outs.append(t_out)

        return student_outs, teacher_outs
    

class DINO_VIT_Tiny(DINO):
    def __init__(self, pretrained=False):
        super().__init__(d_model=192,
                         n_heads=3,
                         n_layers=12,
                         pretrained=pretrained)
        

class DINO_VIT_Small(DINO):
    def __init__(self, pretrained=False):
        super().__init__(d_model=384,
                         n_heads=6,
                         n_layers=12,
                         pretrained=pretrained)
        

class DINO_VIT_Base(DINO):
    def __init__(self, pretrained=False):
        super().__init__(d_model=768,
                         n_heads=12,
                         n_layers=12,
                         pretrained=pretrained)