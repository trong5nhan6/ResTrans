import torch
import torch.nn as nn
import torchvision.models as models


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

        else:
            raise ValueError("Unsupported model")

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)        
        return self.classifier(feat)