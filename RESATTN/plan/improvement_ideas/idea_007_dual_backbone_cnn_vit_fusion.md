# Idea 007 — Dual Backbone Fusion: CNN + ViT Complementary Experts

**Priority**: P3 (Long-term, High Research Value)  
**Complexity**: High  
**Compute Cost**: Large (2× per forward pass)

---

## Motivation

CNN and ViT capture fundamentally different features:
- **CNN (ConvNeXt)**: Local texture, edge patterns, color gradients. Strong for BCC (pearly texture), AKIEC (scale patterns), NV (uniform texture).
- **ViT (ViT-B/16 + ResAttn)**: Global structure, long-range relationships, shape patterns. Strong for MEL (asymmetry, irregular borders), DF (fibrous pattern across large area), VASC (vessel patterns).

The best single model is ConvNeXt at 86.1%. The best ViT model is 79.2% (vitb16_resattn @Ep10). If these are complementary, fusion could reach 88-90%.

---

## Current Problem

We have two expert models trained separately, but no fusion mechanism. Their predictions are currently compared, not combined.

---

## Hypothesis

CNN and ViT are architecturally complementary — they make different types of errors. Fusing their features (at the intermediate level, not just prediction level) will capture both local and global information simultaneously.

Evidence from literature:
- TransFuse (MICCAI 2021): CNN + Transformer fusion → +3-4% over each alone on skin lesion segmentation
- CoAtNet: Convolution + Transformer → SOTA on ImageNet at similar compute
- EfficientDet: FPN-style fusion → consistent improvements on detection

---

## Expected Improvement

- Best single model: ConvNeXt 86.1%
- Expected fusion: 88-91%
- Research novelty: Medium-High (ResAttn-guided dual backbone fusion is novel)

---

## Risk

Medium. Dual backbone:
- Doubles compute
- More complex training (potential gradient interference)
- Risk of one backbone dominating (mitigated by gated fusion)

---

## Implementation Plan

### Architecture Design

```
Input Image
    ├── ConvNeXt-B backbone → [B, 1024] (global avg pool)
    │   └── Stage features: [128, 256, 512, 1024]
    │
    └── ViTB16_AttnRes backbone → [B, 768] (CLS token)
        └── Layer features: [768] × 12 layers

Cross-Attention Fusion Module
    ↓
Final Classifier (7 classes)
```

### Cross-Attention Fusion

```python
class CrossModalFusion(nn.Module):
    """
    Fuses CNN (ConvNeXt) and ViT (ViTB16_AttnRes) features via cross-attention.
    
    CNN features: local, spatial [B, C_cnn]
    ViT features: global, sequential [B, C_vit]
    """
    def __init__(self, cnn_dim=1024, vit_dim=768, fusion_dim=512, num_heads=8):
        super().__init__()
        
        # Project both to same fusion dimension
        self.cnn_proj = nn.Linear(cnn_dim, fusion_dim)
        self.vit_proj = nn.Linear(vit_dim, fusion_dim)
        
        # Cross-attention: CNN features attend to ViT features
        self.cross_attn_cnn2vit = nn.MultiheadAttention(
            fusion_dim, num_heads, batch_first=True
        )
        
        # Cross-attention: ViT features attend to CNN features  
        self.cross_attn_vit2cnn = nn.MultiheadAttention(
            fusion_dim, num_heads, batch_first=True
        )
        
        # Gate to control how much fusion is used
        self.gate_cnn = nn.Parameter(torch.zeros(1))
        self.gate_vit = nn.Parameter(torch.zeros(1))
        
        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
    def forward(self, cnn_feat, vit_feat):
        """
        cnn_feat: [B, cnn_dim]
        vit_feat: [B, vit_dim]
        """
        # Project to fusion space
        q_cnn = self.cnn_proj(cnn_feat).unsqueeze(1)  # [B, 1, D]
        q_vit = self.vit_proj(vit_feat).unsqueeze(1)  # [B, 1, D]
        
        # CNN queries ViT (CNN asks: "what global context does ViT see?")
        cnn_from_vit, _ = self.cross_attn_cnn2vit(q_cnn, q_vit, q_vit)
        cnn_from_vit = cnn_from_vit.squeeze(1)  # [B, D]
        
        # ViT queries CNN (ViT asks: "what local texture does CNN see?")
        vit_from_cnn, _ = self.cross_attn_vit2cnn(q_vit, q_cnn, q_cnn)
        vit_from_cnn = vit_from_cnn.squeeze(1)  # [B, D]
        
        # Gated residual fusion
        cnn_fused = q_cnn.squeeze(1) + self.gate_cnn * (cnn_from_vit - q_cnn.squeeze(1))
        vit_fused = q_vit.squeeze(1) + self.gate_vit * (vit_from_cnn - q_vit.squeeze(1))
        
        # Concatenate and project
        combined = torch.cat([cnn_fused, vit_fused], dim=1)  # [B, 2D]
        return self.fusion_mlp(combined)  # [B, 512]


class DualBackboneModel(nn.Module):
    """
    Dual backbone: ConvNeXt (CNN expert) + ViTB16_AttnRes (Transformer expert).
    """
    def __init__(self, num_classes=7, freeze_cnn=False, freeze_vit=False):
        super().__init__()
        
        # Load pretrained ConvNeXt (can load from checkpoint)
        from model.conv_resattn import ConvNeXt_AttnRes
        from model.vitb16_resattn import ViTB16_AttnRes
        
        # ===== CNN backbone: ConvNeXt-B =====
        self.cnn_backbone = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
        # Remove classifier, get 1024-dim feature
        self.cnn_backbone.classifier[2] = nn.Identity()
        
        if freeze_cnn:
            for p in self.cnn_backbone.parameters():
                p.requires_grad = False
        
        # ===== ViT backbone: ViTB16 + AttnRes =====
        self.vit_backbone = ViTB16_AttnRes(block_size=4, num_classes=num_classes)
        # Remove its head, expose features
        vit_dim = 768
        self.vit_head_removed = True
        
        if freeze_vit:
            for p in self.vit_backbone.parameters():
                p.requires_grad = False
        
        # ===== Cross-modal fusion =====
        self.fusion = CrossModalFusion(
            cnn_dim=1024, vit_dim=vit_dim, fusion_dim=512, num_heads=8
        )
        
        # ===== Final head =====
        self.classifier = CosineClassifier(512, num_classes, scale=30.0)
    
    def get_vit_features(self, x):
        """Extract features from ViT backbone without final head."""
        # Bypass the head
        x = self.vit_backbone.vit._process_input(x)
        n = x.shape[0]
        cls_token = self.vit_backbone.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit_backbone.vit.encoder.pos_embedding
        x = self.vit_backbone.vit.encoder.dropout(x)
        
        blocks = [x]
        partial = x
        for blk in self.vit_backbone.vit.encoder.layers:
            blocks, partial = blk(blocks, partial)
        
        x = self.vit_backbone.vit.encoder.ln(partial)
        return x[:, 0]  # CLS token [B, 768]
    
    def forward(self, x):
        # ===== CNN features =====
        cnn_feat = self.cnn_backbone(x)  # [B, 1024]
        
        # ===== ViT features =====
        vit_feat = self.get_vit_features(x)  # [B, 768]
        
        # ===== Cross-modal fusion =====
        fused = self.fusion(cnn_feat, vit_feat)  # [B, 512]
        
        # ===== Classification =====
        return self.classifier(fused)  # [B, 7]
```

### Training Strategy

```
Step 1: Train ConvNeXt alone (100 epochs) → save checkpoint
Step 2: Train ViTB16_AttnRes alone with LLRD (100 epochs) → save checkpoint
Step 3: Load both checkpoints as initialization
Step 4: Train DualBackboneModel with:
  - Frozen backbones for 10 epochs (train only fusion + classifier)
  - Unfreeze last 4 layers of each backbone for 20 more epochs
  - Full fine-tuning with LLRD for final 30 epochs
```

---

## Files Potentially Affected

- New file: `model/dual_backbone.py`
- `train.py` — add 'dual_backbone' model name
- `config.py` — add dual backbone config

---

## Ablation Plan

| Run | Setup | Expected Test Acc |
|-----|-------|-------------------|
| A | ConvNeXt only | 86.1% (baseline) |
| B | ViTB16_AttnRes only (with LLRD) | ~85% (target) |
| C | Simple average (A+B prediction) | ~87% |
| D | Concat features + linear | ~87% |
| E | CrossModal attention fusion | **~88-90%** |

---

## Success Criteria

- [ ] Test Acc@1 > 87% (surpass both single backbones)
- [ ] Macro F1 > 0.80
- [ ] gate_cnn and gate_vit values are both > 0.1 (both contributing)
- [ ] Compute overhead < 3× single model
- [ ] Attention weights show different patterns for texture-heavy vs shape-heavy classes
