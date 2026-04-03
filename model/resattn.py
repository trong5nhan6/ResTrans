import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# Full Attention Residual
# =========================
class FullAttnRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.zeros(dim))

    def forward(self, prev_values):
        V = torch.stack(prev_values)   # [L, B, T, D]
        K = self.norm(V)

        scores = torch.einsum('d,lbtd->lbt', self.query, K)
        attn = torch.softmax(scores, dim=0)

        h = torch.einsum('lbt,lbtd->btd', attn, V)
        return h


# =========================
# Block
# =========================
class FullAttnResBlock(nn.Module):
    def __init__(self, blk, dim):
        super().__init__()

        self.attn = blk.attn
        self.mlp = blk.mlp
        self.norm1 = blk.norm1
        self.norm2 = blk.norm2

        self.attn_res = FullAttnRes(dim)
        self.mlp_res = FullAttnRes(dim)

    def forward(self, values):
        h = self.attn_res(values)
        attn_out = self.attn(self.norm1(h))
        values.append(attn_out)

        h = self.mlp_res(values)
        mlp_out = self.mlp(self.norm2(h))
        values.append(mlp_out)

        return values


# =========================
# Backbone Wrapper
# =========================
class FullAttnResViT(nn.Module):
    def __init__(self, vit_model, is_dino=True):
        super().__init__()

        self.vit = vit_model
        self.is_dino = is_dino
        dim = vit_model.hidden_dim if not is_dino else vit_model.embed_dim

        # Chọn block list đúng
        if is_dino:
            blk_list = vit_model.blocks
        else:
            blk_list = vit_model.encoder.layers

        self.blocks = nn.ModuleList([FullAttnResBlock(blk, dim) for blk in blk_list])

        # LayerNorm cuối
        self.norm = vit_model.norm if is_dino else nn.LayerNorm(dim)
        self.final_res = FullAttnRes(dim)
        self.dim = dim

    def forward(self, x):
        B, C, H, W = x.shape

        if self.is_dino:
            # DINOv2
            x = self.vit.patch_embed(x)
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.vit.interpolate_pos_encoding(x, W, H)
        else:
            # ViT-B16 torchvision
            x = self.vit._process_input(x)   # patch embed + flatten
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.vit.pos_embed

        values = [x]
        for blk in self.blocks:
            values = blk(values)

        x = self.final_res(values)
        x = self.norm(x)

        return x[:, 0]  # CLS token


# =========================
# CLASSIFIER
# =========================
class FullAttnResClassifier(nn.Module):
    def __init__(self, num_classes, backbone_name="dinov2", freeze_backbone=False):
        super().__init__()

        if backbone_name == "dinov2":
            vit = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14"
            )
            is_dino = True

        elif backbone_name == "vitb16":
            vit = models.vit_b_16(pretrained=False)
            is_dino = False
        else:
            raise ValueError("backbone_name must be 'dinov2' or 'vitb16'")

        self.backbone = FullAttnResViT(vit, is_dino=is_dino)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone.dim),
            nn.Linear(self.backbone.dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)