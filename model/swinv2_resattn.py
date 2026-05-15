"""
SwinTransformerV2 + Block Attention Residuals  (pure torchvision, no timm)
==========================================================================
Loads ImageNet-1k pretrained SwinV2 weights from torchvision, then wraps
every SwinTransformerBlockV2 with Block AttnRes (arXiv 2603.15031).

Key differences from ViT:
  - Swin uses LOCAL window attention  → each depth level has genuinely
    different local features → depth-wise selective aggregation is useful.
  - PostNorm (SwinV2 style): norm(attn(h)), not attn(norm(h)).
  - HWC feature layout [B, H, W, C] instead of ViT's [B, T, D].
  - Per-stage AttnRes: blocks list resets at PatchMerging boundaries
    to avoid the channel-dimension mismatch between stages.

Expected input: 256×256 (SwinV2 pretraining resolution for window_size=8).
Update get_transform(..., img_size=256) in utils.py when using this model.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    swin_v2_t, Swin_V2_T_Weights,
    swin_v2_s, Swin_V2_S_Weights,
    swin_v2_b, Swin_V2_B_Weights,
)


# ── RMSNorm ───────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


# ── Block AttnRes (spatial-aware) ─────────────────────────────────────────────
def block_attn_res(
    blocks: list,
    partial: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """Depth-wise softmax attention over completed blocks + current partial sum.

    Handles both sequence [B, T, D] and spatial [B, H, W, D] tensors by
    flattening intermediate dims, computing attention, then restoring shape.

    proj : Linear(D, 1, bias=False)  — learned pseudo-query w_l
    norm : RMSNorm(D)                — normalises keys to prevent mag bias
    """
    V    = torch.stack(blocks + [partial])      # [N+1, B, ..., D]
    orig = V.shape                              # (N+1, B, *spatial, D)
    B, D = orig[1], orig[-1]

    V_flat = V.view(orig[0], B, -1, D)         # [N+1, B, T, D]
    K      = norm(V_flat)
    w      = proj.weight.squeeze()             # [D]

    logits = torch.einsum("d, n b t d -> n b t", w, K)
    attn   = logits.softmax(dim=0)
    h      = torch.einsum("n b t, n b t d -> b t d", attn, V_flat)

    return h.view(B, *orig[2:])                # restore spatial shape


# ── AttnRes wrapper for one SwinTransformerBlockV2 ────────────────────────────
class AttnResSwinV2Block(nn.Module):
    """Replaces both skip-connections of a SwinTransformerBlockV2 with AttnRes.

    SwinV2 uses *post-norm*:
        original : x  +  stoch(norm1(attn(x)))   +  stoch(norm2(mlp(x)))
        AttnRes  : p  +  stoch(norm1(attn(h)))   +  stoch(norm2(mlp(h)))
    where h = block_attn_res(blocks, p, ...) and p is the intra-block partial sum.
    """

    def __init__(
        self,
        swin_block: nn.Module,
        dim: int,
        block_size: int,
        layer_number: int,          # 1-indexed within its stage
    ):
        super().__init__()

        # ── reuse pretrained components ───────────────────────────────────────
        self.attn             = swin_block.attn
        self.norm1            = swin_block.norm1
        self.mlp              = swin_block.mlp
        self.norm2            = swin_block.norm2
        self.stochastic_depth = swin_block.stochastic_depth

        # ── new AttnRes parameters (zero-init = uniform weights at t=0) ──────
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        self.mlp_res_proj  = nn.Linear(dim, 1, bias=False)
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_norm  = RMSNorm(dim)

        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)

        self.block_size   = block_size
        self.layer_number = layer_number

    def forward(self, blocks: list, hidden_states: torch.Tensor):
        """
        blocks       : list of [B, H, W, C] completed block tensors
        hidden_states: [B, H, W, C] current intra-block partial sum
        Returns      : (blocks, partial)
        """
        partial = hidden_states

        # ── AttnRes aggregation before window-attention ───────────────────────
        h = block_attn_res(blocks, partial, self.attn_res_proj, self.attn_res_norm)

        # ── block boundary: save partial sum and start fresh ──────────────────
        if self.layer_number % (self.block_size // 2) == 0:
            blocks.append(partial)
            partial = None

        # ── SwinV2 window-attention (post-norm, no shortcut) ──────────────────
        # ShiftedWindowAttentionV2 handles window partition/shift internally.
        attn_out = self.stochastic_depth(self.norm1(self.attn(h)))
        partial  = attn_out if partial is None else partial + attn_out

        # ── AttnRes aggregation before MLP ────────────────────────────────────
        h = block_attn_res(blocks, partial, self.mlp_res_proj, self.mlp_res_norm)

        # ── SwinV2 MLP (post-norm, no shortcut) ──────────────────────────────
        mlp_out = self.stochastic_depth(self.norm2(self.mlp(h)))
        partial = partial + mlp_out

        return blocks, partial


# ── Full model ─────────────────────────────────────────────────────────────────
class SwinV2_AttnRes(nn.Module):
    """SwinTransformerV2 backbone with Block Attention Residuals.

    Args:
        variant    : 't' = Tiny (28M), 's' = Small (50M), 'b' = Base (88M)
        block_size : AttnRes block granularity; block boundary every
                     block_size//2 layers within a stage.  Default 4 → every 2.
        num_classes: number of output classes
        pretrained : if True, loads ImageNet-1k weights from torchvision

    Input: 256 × 256 RGB images (required by window_size=8 pretraining).
    """

    _VARIANTS = {
        #           builder   weights                       last_dim
        "t": (swin_v2_t, Swin_V2_T_Weights.IMAGENET1K_V1,  768),
        "s": (swin_v2_s, Swin_V2_S_Weights.IMAGENET1K_V1,  768),
        "b": (swin_v2_b, Swin_V2_B_Weights.IMAGENET1K_V1, 1024),
    }

    def __init__(
        self,
        variant: str = "b",
        block_size: int = 4,
        num_classes: int = 10,
        pretrained: bool = True,
    ):
        super().__init__()
        assert variant in self._VARIANTS, f"variant must be one of {list(self._VARIANTS)}"

        builder, weights_enum, last_dim = self._VARIANTS[variant]
        base = builder(weights=weights_enum if pretrained else None)
        self.block_size = block_size

        # ── patch embedding (features[0]) ─────────────────────────────────────
        # Sequential: Conv2d → Permute([0,2,3,1]) → LayerNorm
        # Output: [B, H/4, W/4, embed_dim]  (HWC format)
        self.patch_embed = base.features[0]

        # ── stages & patch-merging ────────────────────────────────────────────
        # torchvision features layout (all SwinV2 variants):
        # idx: 0           1       2      3       4      5       6      7
        #      patch_embed stage0  merge  stage1  merge  stage2  merge  stage3
        self.stages       = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for s_idx, feat_idx in enumerate([1, 3, 5, 7]):
            stage_seq = base.features[feat_idx]              # nn.Sequential of blocks
            dim       = stage_seq[0].norm1.normalized_shape[0]

            self.stages.append(nn.ModuleList([
                AttnResSwinV2Block(blk, dim, block_size, i + 1)
                for i, blk in enumerate(stage_seq)
            ]))

        for feat_idx in [2, 4, 6]:
            self.downsamplers.append(base.features[feat_idx])  # PatchMerging

        # ── head ─────────────────────────────────────────────────────────────
        self.norm = base.norm                            # final LayerNorm
        self.head = nn.Linear(last_dim, num_classes)

    # ── param groups compatible with train.py ────────────────────────────────
    def get_param_groups(
        self,
        lr_base: float = 1e-4,
        lr_attn_res_norm: float = 1e-3,
        lr_mlp_res_norm: float = 1e-3,
        lr_head: float = 1e-3,
    ) -> list:
        attn_res_norm_params, mlp_res_norm_params, head_params, other_params = [], [], [], []
        special_ids: set = set()

        for stage in self.stages:
            for blk in stage:
                for p in blk.attn_res_norm.parameters():
                    attn_res_norm_params.append(p)
                    special_ids.add(id(p))
                for p in blk.mlp_res_norm.parameters():
                    mlp_res_norm_params.append(p)
                    special_ids.add(id(p))

        for p in self.head.parameters():
            head_params.append(p)
            special_ids.add(id(p))

        for p in self.parameters():
            if id(p) not in special_ids:
                other_params.append(p)

        return [
            {"params": other_params,          "lr": lr_base},
            {"params": attn_res_norm_params,  "lr": lr_attn_res_norm},
            {"params": mlp_res_norm_params,   "lr": lr_mlp_res_norm},
            {"params": head_params,           "lr": lr_head},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── patch embedding ───────────────────────────────────────────────────
        x = self.patch_embed(x)         # [B, H/4, W/4, C]

        # ── stages with per-stage Block AttnRes ───────────────────────────────
        for s_idx, stage in enumerate(self.stages):
            blocks  = [x]               # b_0 = stage input (token embedding for this stage)
            partial = x

            for blk in stage:
                blocks, partial = blk(blocks, partial)

            x = partial

            if s_idx < len(self.downsamplers):
                x = self.downsamplers[s_idx](x)     # PatchMerging: halves HW, doubles C

        # ── final norm + global average pool + head ───────────────────────────
        x = self.norm(x)                # [B, H', W', C']
        x = x.mean(dim=(1, 2))          # [B, C']  — GAP over spatial dims
        x = self.head(x)                # [B, num_classes]
        return x


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    model = SwinV2_AttnRes(variant="b", block_size=4, num_classes=7, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    print("Input :", x.shape)
    print("Output:", out.shape)

    pg = model.get_param_groups()
    for i, g in enumerate(pg):
        n = sum(p.numel() for p in g["params"])
        print(f"Group {i}: {len(g['params']):3d} tensors  {n / 1e6:.2f}M params  lr={g['lr']}")
    print(f"Total : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
