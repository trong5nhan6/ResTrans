import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class CosineClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        self.scale = scale

    def forward(self, x):
        # normalize feature
        x = F.normalize(x, dim=1)

        # normalize weight
        w = F.normalize(self.weight, dim=1)

        # cosine similarity
        logits = torch.matmul(x, w.t())

        # scale
        return logits * self.scale

class DinoV2Backbone(nn.Module):
    def __init__(self, train_last_n_blocks=4):
        super().__init__()

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14"
        )

        self.embed_dim = self.model.embed_dim

        # ===== Freeze toàn bộ =====
        for p in self.model.parameters():
            p.requires_grad = False

        # ===== Unfreeze N block cuối =====
        if train_last_n_blocks > 0:
            for blk in self.model.blocks[-train_last_n_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # mở luôn norm cuối
        for p in self.model.norm.parameters():
            p.requires_grad = True

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.interpolate_pos_encoding(x, W, H)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        return x[:, 1:].mean(dim=1)

class BaseModel(nn.Module):
    def __init__(self, model_name="resnet152", num_classes=3, pretrained=True):
        super().__init__()

        self.model_name = model_name

        if model_name == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet152(weights=weights)
            in_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_name == "convnext":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_base(weights=weights)
            in_dim = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()

        elif model_name == "vit":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()

        elif model_name == "swinv2":
            weights = models.Swin_V2_B_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.swin_v2_b(weights=weights)
            in_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif model_name == "dinov2":
            self.backbone = DinoV2Backbone(train_last_n_blocks=4)
            in_dim = self.backbone.embed_dim

        else:
            raise ValueError("Unsupported model")

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.head = CosineClassifier(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)        
        feat = self.classifier(feat)
        feat = self.head(feat)
        return feat   