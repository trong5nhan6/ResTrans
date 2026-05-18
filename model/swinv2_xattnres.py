"""
SwinTransformerV2 + XAttnRes  (spatial, cross-stage)
=====================================================
Applies XAttnRes (arXiv 2604.03297) ideas to SwinV2 classification.

Key differences from swinv2_resattn.py:

  1. Per-position attention: operates natively on [B, H, W, C] — no
     flatten/unflatten trick. Attention computed independently per (h,w):
         l_n[b,h,w]   = w · RMSNorm(V_n[b,h,w,:])
         α[b,h,w]     = softmax(l, dim=depth)
         out[b,h,w,:] = Σ_n  α_n[b,h,w] · V_n[b,h,w,:]

  2. Cross-stage history pool: after each stage its output is appended to ℋ.
     Before stage s > 0, all history features are spatially aligned
     (resize + 1×1 conv, Eq.4 of XAttnRes paper) to (H_s, W_s, C_s) then
     aggregated via SpatialAttnRes, injecting multi-scale context.

  3. Intra-stage AttnRes (same block-boundary logic as swinv2_resattn.py)
     now uses SpatialAttnRes instead of the old flatten-based aggregation.

Architecture:
  patch_embed → [Stage 0] → ℋ[0] → PatchMerge
    → CrossAttnRes(ℋ[0])          → [Stage 1] → ℋ[1] → PatchMerge
    → CrossAttnRes(ℋ[0..1])       → [Stage 2] → ℋ[2] → PatchMerge
    → CrossAttnRes(ℋ[0..2])       → [Stage 3]
    → LayerNorm → GAP → Head

Input: 256×256 RGB (required by SwinV2 window_size=8 pretraining).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ── Per-position SpatialAttnRes ───────────────────────────────────────────────
class SpatialAttnRes(nn.Module):
    """Attention residual that operates natively on [B, H, W, C].

    Unlike the flatten trick in swinv2_resattn.py, attention weights are
    computed independently at each spatial position (h, w), so local
    features at different positions can attend differently — directly
    implementing the per-position formulation of XAttnRes (Eq. 6–8).

    proj : Linear(C, 1, bias=False)  — learned pseudo-query  w ∈ ℝᶜ
    norm : RMSNorm(C)                — key normalisation
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)
        self.norm = RMSNorm(dim)
        nn.init.zeros_(self.proj.weight)   # uniform weights at init

    def forward(self, history: list, current: torch.Tensor) -> torch.Tensor:
        """
        history : list of [B, H, W, C]  (same res/channels as current)
        current : [B, H, W, C]
        returns : [B, H, W, C]
        """
        V      = torch.stack(history + [current])                     # [N+1, B, H, W, C]
        K      = self.norm(V)                                         # [N+1, B, H, W, C]
        w      = self.proj.weight.squeeze()                           # [C]
        logits = torch.einsum("c, n b h w c -> n b h w", w, K)       # [N+1, B, H, W]
        attn   = logits.softmax(dim=0)                                # [N+1, B, H, W]
        out    = torch.einsum("n b h w, n b h w c -> b h w c", attn, V)
        return out


# ── Spatial aligner (Eq.4 of XAttnRes) ───────────────────────────────────────
class StageAligner(nn.Module):
    """Aligns one history feature map to a target resolution + channel count.

    ĥₖ = Conv₁ₓ₁⁽ᵏ⁾(Resize(hₖ, (H', W')))

    HWC ↔ CHW conversions are handled internally.
    When src and target resolutions already match, interpolation is skipped.
    """

    def __init__(self, src_dim: int, tgt_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(src_dim, tgt_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, tgt_hw: tuple) -> torch.Tensor:
        # x: [B, H_src, W_src, C_src]
        x = x.permute(0, 3, 1, 2).contiguous()              # → [B, C_src, H, W]
        if (x.shape[2], x.shape[3]) != tgt_hw:
            x = F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)
        x = self.conv(x)                                     # → [B, C_tgt, H', W']
        return x.permute(0, 2, 3, 1).contiguous()           # → [B, H', W', C_tgt]


# ── Cross-stage XAttnRes module ───────────────────────────────────────────────
class CrossStageXAttnRes(nn.Module):
    """Aggregates the global history pool at a stage boundary.

    For stage s with input [B, H_s, W_s, C_s]:
      - Each ℋ[k] is aligned to (H_s, W_s, C_s) via StageAligner
      - SpatialAttnRes aggregates [aligned_ℋ[0], ..., aligned_ℋ[s-1], x]
        with per-position attention weights

    history_dims : list of source channel counts  [C_0, ..., C_{s-1}]
    tgt_dim      : channel count of stage s  (C_s)
    """

    def __init__(self, history_dims: list, tgt_dim: int):
        super().__init__()
        self.aligners = nn.ModuleList([
            StageAligner(src_dim, tgt_dim) for src_dim in history_dims
        ])
        self.attn_res = SpatialAttnRes(tgt_dim)

    def forward(self, history: list, x: torch.Tensor) -> torch.Tensor:
        """
        history : list of [B, H_k, W_k, C_k] for k < current stage
        x       : [B, H_s, W_s, C_s]
        returns : [B, H_s, W_s, C_s]
        """
        tgt_hw  = (x.shape[1], x.shape[2])
        aligned = [al(h, tgt_hw) for al, h in zip(self.aligners, history)]
        return self.attn_res(aligned, x)


# ── Intra-stage block wrapper ─────────────────────────────────────────────────
class SwinBlockXAttn(nn.Module):
    """Wraps one SwinTransformerBlockV2 with intra-stage SpatialAttnRes.

    Keeps the same block-boundary logic as swinv2_resattn.py but replaces
    the flatten-based block_attn_res with two SpatialAttnRes modules:
    one before window-attention, one before MLP.
    """

    def __init__(
        self,
        swin_block: nn.Module,
        dim: int,
        block_size: int,
        layer_number: int,          # 1-indexed within stage
    ):
        super().__init__()
        self.attn             = swin_block.attn
        self.norm1            = swin_block.norm1
        self.mlp              = swin_block.mlp
        self.norm2            = swin_block.norm2
        self.stochastic_depth = swin_block.stochastic_depth

        self.attn_res  = SpatialAttnRes(dim)
        self.mlp_res   = SpatialAttnRes(dim)

        self.block_size   = block_size
        self.layer_number = layer_number

    def forward(self, blocks: list, partial: torch.Tensor):
        """
        blocks  : completed super-block partial sums  [B, H, W, C]
        partial : current accumulator                 [B, H, W, C]
        """
        # aggregate before window-attention
        h = self.attn_res(blocks, partial)

        # block boundary: save partial and start fresh
        if self.layer_number % (self.block_size // 2) == 0:
            blocks.append(partial)
            partial = None

        # SwinV2 window-attention (post-norm, no plain shortcut)
        attn_out = self.stochastic_depth(self.norm1(self.attn(h)))
        partial  = attn_out if partial is None else partial + attn_out

        # aggregate before MLP
        h = self.mlp_res(blocks, partial)

        # SwinV2 MLP (post-norm, no plain shortcut)
        mlp_out = self.stochastic_depth(self.norm2(self.mlp(h)))
        partial = partial + mlp_out

        return blocks, partial


# ── Full model ────────────────────────────────────────────────────────────────
class SwinV2_XAttnRes(nn.Module):
    """SwinV2 with per-position intra-stage + cross-stage XAttnRes.

    Args:
        variant    : 't' Tiny 28M | 's' Small 50M | 'b' Base 88M
        block_size : intra-stage super-block size; boundary every
                     block_size//2 layers (default 4 → every 2 layers)
        num_classes: output classes
        pretrained : load ImageNet-1k weights from torchvision

    Input: 256×256 RGB.
    """

    _VARIANTS = {
        "t": (swin_v2_t, Swin_V2_T_Weights.IMAGENET1K_V1),
        "s": (swin_v2_s, Swin_V2_S_Weights.IMAGENET1K_V1),
        "b": (swin_v2_b, Swin_V2_B_Weights.IMAGENET1K_V1),
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

        builder, weights_enum = self._VARIANTS[variant]
        base = builder(weights=weights_enum if pretrained else None)
        self.block_size = block_size

        # ── patch embedding ───────────────────────────────────────────────────
        self.patch_embed = base.features[0]

        # ── derive stage dims from pretrained weights ─────────────────────────
        # torchvision layout: idx 1,3,5,7 = stages; idx 2,4,6 = PatchMerging
        stage_dims = [
            base.features[fi][0].norm1.normalized_shape[0]
            for fi in [1, 3, 5, 7]
        ]
        self.stage_dims = stage_dims

        # ── stages: wrap each SwinBlock with SpatialAttnRes ───────────────────
        self.stages = nn.ModuleList()
        for feat_idx in [1, 3, 5, 7]:
            stage_seq = base.features[feat_idx]
            dim       = stage_seq[0].norm1.normalized_shape[0]
            self.stages.append(nn.ModuleList([
                SwinBlockXAttn(blk, dim, block_size, i + 1)
                for i, blk in enumerate(stage_seq)
            ]))

        # ── patch merging ─────────────────────────────────────────────────────
        self.downsamplers = nn.ModuleList([
            base.features[fi] for fi in [2, 4, 6]
        ])

        # ── cross-stage XAttnRes ──────────────────────────────────────────────
        # cross_stage[i] is applied at the start of stage (i+1):
        #   cross_stage[0]: history=[dims[0]]         → target dims[1]
        #   cross_stage[1]: history=[dims[0],dims[1]] → target dims[2]
        #   cross_stage[2]: history=[dims[0..2]]      → target dims[3]
        self.cross_stage = nn.ModuleList([
            CrossStageXAttnRes(stage_dims[:s], stage_dims[s])
            for s in range(1, 4)
        ])

        # ── head ──────────────────────────────────────────────────────────────
        self.norm = base.norm
        self.head = nn.Linear(stage_dims[-1], num_classes)

    # ── param groups compatible with train.py ────────────────────────────────
    def get_param_groups(
        self,
        lr_base: float = 1e-4,
        lr_attn_res_norm: float = 1e-3,
        lr_mlp_res_norm: float = 1e-3,
        lr_head: float = 1e-3,
    ) -> list:
        attn_res_norm_p, mlp_res_norm_p, head_p = [], [], []
        special_ids: set = set()

        # intra-stage norms
        for stage in self.stages:
            for blk in stage:
                for p in blk.attn_res.norm.parameters():
                    attn_res_norm_p.append(p); special_ids.add(id(p))
                for p in blk.mlp_res.norm.parameters():
                    mlp_res_norm_p.append(p); special_ids.add(id(p))

        # cross-stage norms
        for cs in self.cross_stage:
            for p in cs.attn_res.norm.parameters():
                attn_res_norm_p.append(p); special_ids.add(id(p))

        # head
        for p in self.head.parameters():
            head_p.append(p); special_ids.add(id(p))

        other_p = [p for p in self.parameters() if id(p) not in special_ids]

        return [
            {"params": other_p,          "lr": lr_base},
            {"params": attn_res_norm_p,  "lr": lr_attn_res_norm},
            {"params": mlp_res_norm_p,   "lr": lr_mlp_res_norm},
            {"params": head_p,           "lr": lr_head},
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)           # [B, H/4, W/4, C₀]

        history = []                      # global history pool ℋ

        for s_idx, stage in enumerate(self.stages):
            # inject multi-scale context from previous stage outputs
            if s_idx > 0:
                x = self.cross_stage[s_idx - 1](history, x)

            # intra-stage block AttnRes
            blocks  = [x]
            partial = x
            for blk in stage:
                blocks, partial = blk(blocks, partial)
            x = partial

            history.append(x)             # append to ℋ BEFORE downsampling

            if s_idx < len(self.downsamplers):
                x = self.downsamplers[s_idx](x)   # PatchMerge: ½HW, ×2C

        x = self.norm(x)                  # [B, H', W', C']
        x = x.mean(dim=(1, 2))            # GAP → [B, C']
        x = self.head(x)                  # [B, num_classes]
        return x


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    model = SwinV2_XAttnRes(variant="b", block_size=4, num_classes=7, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    print("Input :", x.shape)
    print("Output:", out.shape)

    pg = model.get_param_groups()
    for i, g in enumerate(pg):
        n = sum(p.numel() for p in g["params"])
        print(f"Group {i}: {len(g['params']):3d} tensors  {n/1e6:.2f}M params  lr={g['lr']}")
    print(f"Total : {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"Stage dims: {model.stage_dims}")
