# Idea 006 — Multi-Scale CLS Token Aggregation from ViT Intermediate Layers

**Priority**: P2  
**Complexity**: Medium  
**Compute Cost**: Low (no additional forward passes)

---

## Motivation

ViT-B/16's 12 transformer layers capture features at progressively higher levels of abstraction:
- Layers 1-4: Low-level features (edges, textures, local patterns)
- Layers 5-8: Mid-level features (lesion borders, color gradients)
- Layers 9-12: High-level semantic features (lesion type, shape)

For skin lesion classification, BOTH low-level texture (BCC: pearly appearance, AKIEC: scaling) AND high-level shape (MEL: asymmetry, NV: round shape) are diagnostically relevant.

The current architecture uses only the **final layer** CLS token. This throws away intermediate diagnostic information.

---

## Current Problem

`vitb16_resattn.py` final output:
```python
x = self.vit.encoder.ln(x)
cls = x[:, 0]   # ← only final layer CLS token
out = self.head(cls)
```

The ResAttn mechanism does aggregate across block states, but only at the hidden state level — not at the final feature level. The CLS token from layer 6 (mid-level features) is never directly exposed to the classifier.

---

## Hypothesis

Aggregating CLS tokens from layers 3, 6, 9, 12 (four scale levels) and fusing them will provide richer multi-scale features, especially for textural classes (BCC, AKIEC) that rely on low-to-mid level features.

---

## Expected Improvement

- ConvNeXt achieves 86% partly due to its hierarchical multi-scale feature design
- Adding multi-scale CLS aggregation to ViT should close this gap
- Expected: +2-3% Test Acc@1

---

## Implementation Plan

### Design 1: Concat + Linear

```python
class MultiScaleCLSHead(nn.Module):
    """
    Extracts CLS tokens from multiple ViT layers and fuses them.
    
    Args:
        dim: hidden dimension (768 for ViT-B/16)
        num_scales: number of intermediate layers to use (default 4)
        num_classes: output classes
        fusion: 'concat', 'attention', or 'mean'
    """
    def __init__(self, dim=768, num_scales=4, num_classes=7, fusion='attention'):
        super().__init__()
        self.fusion = fusion
        
        if fusion == 'concat':
            self.proj = nn.Sequential(
                nn.LayerNorm(dim * num_scales),
                nn.Linear(dim * num_scales, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        
        elif fusion == 'attention':
            # Learn to weight different scales
            self.scale_attn = nn.Sequential(
                nn.Linear(dim, 1),
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 512),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        
        elif fusion == 'mean':
            self.proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
    
    def forward(self, cls_tokens):
        """
        cls_tokens: list of [B, D] tensors from multiple layers
        """
        stacked = torch.stack(cls_tokens, dim=1)  # [B, S, D]
        
        if self.fusion == 'concat':
            flat = stacked.flatten(1)   # [B, S*D]
            return self.proj(flat)
        
        elif self.fusion == 'attention':
            attn_weights = self.scale_attn(stacked).softmax(dim=1)  # [B, S, 1]
            fused = (attn_weights * stacked).sum(dim=1)              # [B, D]
            return self.proj(fused)
        
        elif self.fusion == 'mean':
            fused = stacked.mean(dim=1)  # [B, D]
            return self.proj(fused)
```

### Integration into ViTB16_AttnRes

```python
class ViTB16_AttnRes_MultiScale(nn.Module):
    def __init__(self, block_size=4, num_classes=10, 
                 extract_layers=(2, 5, 8, 11)):
        super().__init__()
        
        self.vit = vit_b_16(weights="IMAGENET1K_V1")
        dim = 768
        self.block_size = block_size
        self.extract_layers = extract_layers  # 0-indexed, which layers to extract CLS
        
        # ... same AttnResBlock setup as before ...
        
        # Multi-scale head
        self.head = MultiScaleCLSHead(
            dim=dim,
            num_scales=len(extract_layers),
            num_classes=num_classes,
            fusion='attention'  # learned scale weighting
        )
        
        # Per-scale normalization
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in extract_layers
        ])
    
    def forward(self, x):
        x = self.vit._process_input(x)
        n = x.shape[0]
        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        
        blocks = [x]
        partial = x
        
        cls_tokens = []
        
        for i, blk in enumerate(self.vit.encoder.layers):
            blocks, partial = blk(blocks, partial)
            
            # Extract CLS token at specified layers
            if i in self.extract_layers:
                scale_idx = self.extract_layers.index(i)
                cls = partial[:, 0]  # CLS token
                cls = self.scale_norms[scale_idx](cls)
                cls_tokens.append(cls)
        
        # Final normalization
        x = self.vit.encoder.ln(partial)
        final_cls = x[:, 0]
        
        # If we're not already including the final layer
        if 11 not in self.extract_layers:
            cls_tokens.append(final_cls)
        
        return self.head(cls_tokens)
```

---

## Design 2: Feature Pyramid Aggregation (FPA)

More principled: project each scale to the same dimension, then apply attention pooling.

```python
class ViTFeaturePyramid(nn.Module):
    """
    FPN-style feature aggregation for ViT intermediate layers.
    Top-down: later (higher-level) features propagated to earlier (lower-level) via addition.
    """
    def __init__(self, dim=768, out_dim=768, num_layers=12, block_size=3):
        super().__init__()
        
        # Scale: extract every block_size layers
        self.pyramid_layers = list(range(block_size-1, num_layers, block_size))  # [2, 5, 8, 11]
        
        # Top-down connections
        self.td_convs = nn.ModuleList([
            nn.Linear(dim, out_dim) for _ in self.pyramid_layers
        ])
        
        # Final aggregation
        self.final_proj = nn.Linear(out_dim, out_dim)
```

---

## Files Potentially Affected

- `model/vitb16_resattn.py` — new class `ViTB16_AttnRes_MultiScale`
- `train.py` — add model name registration
- `config.py` — add `EXTRACT_LAYERS` config

---

## Ablation Plan

| Run | Scale Layers | Fusion | Expected Acc |
|-----|-------------|--------|-------------|
| A | [11] only (current) | linear | 75-79% |
| B | [5, 11] | concat | +1% |
| C | [2, 5, 8, 11] | concat | +2% |
| D | [2, 5, 8, 11] | attention | **+2-3%** (recommended) |
| E | [2, 5, 8, 11] | FPN top-down | +2-3% |

---

## Success Criteria

- [ ] Test Acc@1 ≥ 82% (vs 79% single-scale best)
- [ ] Scale attention weights should NOT collapse to one layer (entropy of weights > 0.5)
- [ ] Macro F1 ≥ 0.70 (especially texture-based classes BCC, AKIEC)
- [ ] No significant increase in training time (< 10%)
