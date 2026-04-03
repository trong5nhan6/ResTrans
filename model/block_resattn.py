import torch
import torch.nn as nn
import torchvision.models as models

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.scale

class BlockAttnRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.query = nn.Parameter(torch.zeros(dim))  # paper init

    def forward(self, blocks, partial):
        """
        blocks: list of [B, T, D]
        partial: [B, T, D]
        """
        if partial is not None:
            V = torch.stack(blocks + [partial])  # [N+1, B, T, D]
        else:
            V = torch.stack(blocks)              # [N, B, T, D]

        K = self.norm(V)

        scores = torch.einsum('d,nbtd->nbt', self.query, K)
        attn = torch.softmax(scores, dim=0)

        h = torch.einsum('nbt,nbtd->btd', attn, V)
        return h

class BlockAttnResLayer(nn.Module):
    def __init__(self, blk, dim, layer_id, block_size):
        super().__init__()

        self.attn = blk.attn
        self.mlp = blk.mlp
        self.norm1 = blk.norm1
        self.norm2 = blk.norm2

        self.attn_res = BlockAttnRes(dim)
        self.mlp_res = BlockAttnRes(dim)

        self.layer_id = layer_id
        self.block_size = block_size  # tính theo (attn+mlp)

    def forward(self, blocks, partial):
        # ===== Pre-Attn AttnRes =====
        h = self.attn_res(blocks, partial)

        attn_out = self.attn(self.norm1(h))
        partial = attn_out if partial is None else partial + attn_out

        # ===== Pre-MLP AttnRes =====
        h = self.mlp_res(blocks, partial)

        mlp_out = self.mlp(self.norm2(h))
        partial = partial + mlp_out

        # ===== Block boundary =====
        if (self.layer_id + 1) % self.block_size == 0:
            blocks.append(partial)
            partial = None

        return blocks, partial

class BlockAttnResViT(nn.Module):
    def __init__(self, vit_model, is_dino=True, block_size=4):
        super().__init__()

        self.vit = vit_model
        self.is_dino = is_dino

        dim = vit_model.embed_dim if is_dino else vit_model.hidden_dim

        if is_dino:
            blk_list = vit_model.blocks
        else:
            blk_list = vit_model.encoder.layers

        self.layers = nn.ModuleList([
            BlockAttnResLayer(blk, dim, i, block_size)
            for i, blk in enumerate(blk_list)
        ])

        self.norm = vit_model.norm if is_dino else nn.LayerNorm(dim)
        self.final_res = BlockAttnRes(dim)
        self.dim = dim

    def forward(self, x):
        B, C, H, W = x.shape

        if self.is_dino:
            x = self.vit.patch_embed(x)
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.vit.interpolate_pos_encoding(x, W, H)
        else:
            x = self.vit._process_input(x)
            cls_token = self.vit.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.vit.pos_embed

        # ===== INIT =====
        blocks = [x]   # b0 = embedding (paper)
        partial = None

        for layer in self.layers:
            blocks, partial = layer(blocks, partial)

        # ===== Final aggregation =====
        final_res = BlockAttnRes(self.dim)
        x = self.final_res(blocks, partial)
        x = self.norm(x)

        return x[:, 0]

# =========================
# CLASSIFIER
# =========================
class BlockAttnResClassifier(nn.Module):
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

        self.backbone = BlockAttnResViT(vit, is_dino=is_dino)

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