# Task 003 — Fix FullAttnRes + Retrain with DINOv2

**Status**: READY (after Task 001)  
**Estimated Time**: 1 hour coding + 4-5 hours training  
**Priority**: P1 — Tests whether the architecture is fundamentally sound

---

## Objective

After fixing the missing residual connections in `resattn.py`, retrain `FullAttnResClassifier` with DINOv2 backbone to determine:
1. Whether the architecture is fundamentally sound (expect 70-80%)
2. Whether DINOv2 features are being properly utilized
3. What performance gap remains vs ConvNeXt (86.1%)

---

## Exact Modules to Modify

**File**: `model/resattn.py`

### Step 1: Fix FullAttnResBlock.forward()

```python
class FullAttnResBlock(nn.Module):
    def __init__(self, blk, dim):
        super().__init__()
        self.attn = blk.attn
        self.mlp = blk.mlp
        self.norm1 = blk.norm1
        self.norm2 = blk.norm2
        self.attn_res = FullAttnRes(dim)
        self.mlp_res = FullAttnRes(dim)
        
        # Gating: init to 0 to preserve pretrained behavior at init
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, values):
        # ===== Step 1: Aggregate from all previous blocks =====
        h_blend = self.attn_res(values)
        
        # ===== Step 2: Gated blend with last output (preserve pretrained at init) =====
        last = values[-1]
        h = last + self.gamma_attn * (h_blend - last)   # gamma=0 → pure pretrained at init
        
        # ===== Step 3: Standard attention (WITH RESIDUAL) =====
        attn_out = self.attn(self.norm1(h))
        out_attn = h + attn_out                          # ← KEY FIX: proper residual
        values.append(out_attn)

        # ===== Step 4: Aggregate again =====
        h2_blend = self.mlp_res(values)
        h2 = out_attn + self.gamma_mlp * (h2_blend - out_attn)
        
        # ===== Step 5: Standard MLP (WITH RESIDUAL) =====
        mlp_out = self.mlp(self.norm2(h2))
        out_mlp = h2 + mlp_out                           # ← KEY FIX: proper residual
        values.append(out_mlp)

        return values
```

### Step 2: Fix FullAttnRes — Add temperature scaling and proper query init

```python
class FullAttnRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # ← FIXED: small random init (not zeros which cause uniform attention forever)
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.temperature = math.sqrt(dim)   # fixed temperature (not learned)

    def forward(self, prev_values):
        V = torch.stack(prev_values)   # [L, B, T, D]
        K = self.norm(V)

        scores = torch.einsum('d,lbtd->lbt', self.query, K) / self.temperature
        attn = torch.softmax(scores, dim=0)

        h = torch.einsum('lbt,lbtd->btd', attn, V)
        return h
```

### Step 3: Fix FullAttnResClassifier head — Add CosineClassifier

```python
class FullAttnResClassifier(nn.Module):
    def __init__(self, num_classes, backbone_name="dinov2", freeze_backbone=False):
        super().__init__()

        if backbone_name == "dinov2":
            vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            is_dino = True
        elif backbone_name == "vitb16":
            vit = models.vit_b_16(pretrained=True)   # ← FIXED: use pretrained=True
            is_dino = False
        
        self.backbone = FullAttnResViT(vit, is_dino=is_dino)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ← IMPROVED: Add bottleneck + CosineClassifier (consistent with best baselines)
        dim = self.backbone.dim
        self.projector = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = CosineClassifier(512, num_classes, scale=30.0)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.projector(feat)
        return self.head(feat)
```

---

## Training Config for This Run

```python
MODEL_NAME = "resattn"
LR = 1e-5              # Lower LR for DINOv2 (it's already very well pretrained)
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50         # DINOv2 should converge faster
BATCH_SIZE = 32

USE_LLRD = False        # DINOv2 uses its own LR handling
FOCAL_LOSS = False      # CrossEntropy with class weights
USE_CUTMIX = True
USE_MIXUP = True
MINORITY_CLASS = None
```

---

## Expected Metrics

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Test Acc@1 @Ep20 | 41.3% | **70-78%** |
| Test F1 | 0.35 | 0.62-0.72 |
| Training Loss @Ep1 | ~2.0 | ~1.6-1.8 |
| Convergence | Slow/none | Fast (DINOv2 features) |

---

## Validation Steps

1. Run 3 epochs and check: loss should be decreasing smoothly, epoch 1 loss ≈ 1.6-2.0
2. If loss > 3.0 at epoch 1: something is still broken (check DINOv2 loading)
3. If val acc doesn't exceed 70% by epoch 10: architecture may need further debugging
4. Compare attention maps: `FullAttnRes.attn` should show non-uniform weights over blocks

---

## Rollback Plan

If DINOv2 fails to load (no internet/hub issue):
1. Switch to `backbone_name="vitb16"` with ImageNet pretrained weights
2. Expected performance will be lower (~65-72%) but still validates the fix
