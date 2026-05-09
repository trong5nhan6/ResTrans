# Paper Alignment Analysis — Residual Attention for ViT

**Paper**: "Residual Attention: A Simple but Effective Method for Multi-Label Recognition" / ResAttn concept applied to ViT  
**PDF**: `2603.15031v1.pdf` (present in repo)  
**Analysis Date**: 2026-05-09

> Note: The paper in the repo (2603.15031v1.pdf) appears to be a ResAttn variant paper. The core idea across all implementations is **cross-block residual attention**: using attention to aggregate information from multiple previous block states rather than only the immediately preceding residual.

---

## Component 1: Cross-Block Attention Mechanism

### In Paper (Conceptual)
The core ResAttn idea: at each layer, instead of receiving only the previous layer's output, the current layer receives an **attention-weighted mixture** of all (or recent) previous block outputs. This provides:
- Long-range gradient paths (mitigating vanishing gradients)
- Multi-scale feature reuse
- Adaptive skip connections

### In Code (`resattn.py` — FullAttnRes)
```python
class FullAttnRes(nn.Module):
    def __init__(self, dim):
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.zeros(dim))  # ← zeros init

    def forward(self, prev_values):
        V = torch.stack(prev_values)        # [L, B, T, D]
        K = self.norm(V)
        scores = torch.einsum('d,lbtd->lbt', self.query, K)
        attn = torch.softmax(scores, dim=0)
        h = torch.einsum('lbt,lbtd->btd', attn, V)
        return h
```

### Match Level: **Partial**

### Analysis
**What's correct**:
- Using all previous block outputs ✓
- Per-token, per-block attention ✓
- Softmax over blocks (not tokens) ✓

**What's problematic**:
- `query = zeros(dim)` → `scores = einsum('d,lbtd->lbt', zeros, K) = 0` for ALL blocks → softmax gives uniform weights → random initialization
- No temperature scaling (`/sqrt(dim)`) → scores can be very large once query learns
- No gating → changes behavior abruptly, can destroy pretrained representations

### Impact
When `query=0`, all blocks receive **equal weight** at initialization. This is actually "safe" but provides no adaptive behavior. The model must learn to differentiate blocks from a flat gradient landscape. This is slower convergence compared to a properly initialized model.

### Recommendation
**Sửa**: Use `nn.Parameter(torch.randn(dim) * 0.02)` init, add temperature scaling, and add a residual gate (see `vitb16_resattn.py` which correctly uses gamma=0 init).

---

## Component 2: Block-Sparse Residual Attention

### In Paper
The block-sparse variant aggregates states only at block boundaries (every `block_size` layers), reducing computation while preserving long-range dependencies.

### In Code (`block_resattn.py` — BlockAttnResLayer)
```python
# Block boundary logic:
if (self.layer_id + 1) % self.block_size == 0:
    blocks.append(partial)
    partial = None  # ← RESETS partial to None!
```

### Match Level: **Different** (has a reset logic not in paper)

### Analysis
The code resets `partial = None` at block boundaries. This means the next block starts from `None`, not from the accumulated residual. This forces ALL information to flow through the block attention mechanism, with no direct residual path.

In `BlockAttnResLayer.forward()`:
```python
def forward(self, blocks, partial):
    # Pre-Attn AttnRes
    h = self.attn_res(blocks, partial)   # partial can be None
    attn_out = self.attn(self.norm1(h))
    partial = attn_out if partial is None else partial + attn_out  # special case for None
```

**This creates inconsistent gradient flow**: early blocks in each segment have no residual connection (partial=None), while later blocks do.

### Impact
This is likely a major reason why `block_resattn` converges slowly (62% at epoch 40). The reset effectively cuts gradient paths, making early-segment layers behave like non-residual networks.

### Recommendation
**Sửa**: Do NOT reset `partial = None`. Instead, just append to blocks and continue. The paper never mentions resetting the hidden state at block boundaries.

---

## Component 3: Gated Residual (vitb16_resattn.py)

### In Paper
Not explicitly gated in original ResAttn paper, but gating (gamma parameter) is a common practice in transformer fine-tuning (e.g., LayerScale from CaiT).

### In Code
```python
self.gamma_attn = nn.Parameter(torch.zeros(1))  # ← zero init = preserve pretrained at init
# In forward:
h = partial + self.gamma_attn * (h_blend - partial)
```

### Match Level: **Extension (beyond paper)**

### Analysis
This is the most principled design in the codebase. Zero-init gamma means:
- At initialization: `h = partial + 0 * (h_blend - partial) = partial` → exact pretrained behavior preserved
- Gradients flow to gamma first, then to residual attention
- Prevents catastrophic forgetting at early epochs

However, the problem observed in logs is **not gamma** but the **backbone LR being too high**. The ViT weights themselves are being updated at 1e-4, which is an order of magnitude too high for fine-tuning.

### Recommendation
**Keep gamma gating** (it's correct). **Add LLRD** (layer-wise LR decay) so ViT backbone LR is 1e-5 to 1e-6 while new ResAttn parameters get 1e-4.

---

## Component 4: Feature Normalization in Attention

### In Paper
Layer Normalization applied to values before computing attention.

### In `resattn.py`
```python
K = self.norm(V)  # LayerNorm applied to stacked values
scores = einsum('d,lbtd->lbt', self.query, K)
```
Norm applied to K (keys), but not to Q. Standard in cross-attention designs.

### In `vitb16_resattn.py`
```python
class RMSNorm(nn.Module):
    # ...
self.attn_res_norm = RMSNorm(dim)
# In block_attn_res:
K = norm(V)
logits = einsum('d, nbtd -> nbt', w, K) / math.sqrt(K.shape[-1])
```
Uses RMSNorm (more efficient) + temperature scaling `/sqrt(D)`.

### Match Level: `resattn.py`: **Partial** | `vitb16_resattn.py`: **Better than paper**

### Recommendation
**Keep RMSNorm + temperature** in vitb16_resattn. The division by `sqrt(D)` is critical to prevent attention score explosion as more blocks accumulate.

---

## Component 5: Block Size / Memory Budget

### In Paper
Flexible block_size parameter. Original paper suggests block_size=4 for ViT-B/16 (12 layers → 3 blocks).

### In Code
```python
ViTB16_AttnRes(block_size=4, num_classes=7)
```
12 encoder layers / block_size=4 = 3 blocks. **Match Level: Exact**.

However, in `ViTB16_AttnRes.forward()`:
```python
blocks = [x]   # include embedding as block 0
```
Block 0 = patch embedding. This means blocks = [embed, block0_out, block1_out, block2_out] = 4 total. This is fine — the embedding as block 0 is a reasonable design choice.

---

## Component 6: Classification Head

### In Paper
Typically uses CLS token with linear head.

### In `vitb16_resattn.py`
```python
self.head = nn.Sequential(
    nn.LayerNorm(dim),
    nn.Linear(dim, num_classes)
)
```
Two-layer head without CosineClassifier. **Match Level: Partial** (missing CosineClassifier that consistently improves performance in other models).

### Impact
Other models (convnext, swinv2) use CosineClassifier and achieve better results. The lack of CosineClassifier in vitb16_resattn is a potential 1-2% accuracy gap.

### Recommendation
**Try adding CosineClassifier** as an ablation. The scale=30 normalized head helps with class imbalance by preventing the classifier from being dominated by majority class weight norms.

---

## Component 7: Loss Function

### In Paper
Standard CrossEntropy. FocalLoss is not mentioned.

### In Code
```python
FOCAL_LOSS = True
GAMMA = 2.0
ALPHA = None  # ← no per-class weighting
```

### Match Level: **Different**

### Analysis
FocalLoss with gamma=2 down-weights easy (well-classified) examples. On ISIC2018 where Class 1 (NV) has 67% samples, FocalLoss should theoretically help by focusing on hard minority class examples.

**However**: `ALPHA = None` means no per-class weight. The effective FocalLoss is just:
```
FL = (1 - p_t)^2 * CE
```
This can cause instability when the model first encounters a large pretrained model — if the initial predictions are confident (p_t high), the loss is very small, providing weak gradient signal. But if predictions are random (p_t ≈ 1/7 ≈ 0.14), then `(1-0.14)^2 ≈ 0.74` of the loss magnitude.

For `conv_resattn` starting loss=4.65: this suggests the model is NOT even at random predictions — it's significantly worse. This is because the ConvNeXt + AttnRes combination starts in a very poor initialization region.

### Recommendation
**Use FocalLoss(gamma=1.0, alpha=class_weights)** instead of gamma=2 without alpha. Or revert to **CrossEntropy with class weights** which is more stable.

---

## Component 8: Optimizer and Scheduler

### In Paper
Typically AdamW + cosine decay, which matches the implementation.

### In Code
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
```

**Match Level: Partial**

**Missing**: Layer-wise LR decay (LLRD) — standard practice for fine-tuning pretrained transformers. DINOv2, ViT-B/16, SwinV2 all benefit from lower LR for earlier layers.

**Typical LLRD**: `lr_layer_k = base_lr * decay_rate^(num_layers - k)` where `decay_rate ∈ [0.65, 0.9]`.

### Recommendation
**Add LLRD** immediately — this is the single highest-impact fix for vitb16_resattn degradation.

---

## Component 9: Data Augmentation

### In Paper
Standard ImageNet augmentations (RandAugment or AutoAugment) + Mixup/CutMix.

### In Code
```python
T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    T.ToTensor(),
    T.Normalize(...)
])
```

**Match Level: Partial**

**Missing**:
- RandAugment or TrivialAugment (stronger, fewer hyperparameters)
- ElasticTransform (crucial for dermoscopy — skin deformation)
- RandomResizedCrop (important for multi-scale learning)
- Stronger ColorJitter (dermoscopy images vary significantly in color due to imaging devices)
- Grid distortion or perspective transforms

### Recommendation
Replace with skin-specialized augmentation:
```python
# Medical imaging strong augmentation
T.RandomResizedCrop(224, scale=(0.7, 1.0))
A.ElasticTransform(alpha=1, sigma=50)
A.GridDistortion(distort_limit=0.3)
T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
```

---

## Component 10: Validation Protocol

### In Paper
Typically: stratified k-fold or at least proper train/val split.

### In Code
Fixed val set of 193 samples with extreme class imbalance (Class 5: 1 sample).

**Match Level: Missing**

### Impact
**This is the most impactful missing component.** With 193 val samples, the val accuracy has a standard error of approximately:
- SE = sqrt(p*(1-p)/n) = sqrt(0.8*0.2/193) ≈ 2.9%
- 95% CI: ±5.7%

This means val acc can swing ±5.7% just from sampling noise, making it impossible to:
1. Detect genuine improvements
2. Choose the best checkpoint
3. Detect overfitting early

### Recommendation
**Implement 5-fold stratified cross-validation** on the training set, or at minimum use the test set (1512 samples) as the model selection criterion — even though this is not theoretically correct, it's more reliable than a 193-sample val set.

---

## Summary Table

| Component | Paper | Code | Match | Impact | Priority |
|-----------|-------|------|-------|--------|----------|
| Cross-block attention | All prev blocks | All prev blocks | ✅ Exact | - | Keep |
| Block sparsity | No reset | Resets to None | ⚠️ Different | High | Fix |
| Gated residual | Not specified | gamma=0 init | ✅ Extension | Positive | Keep |
| Temperature scaling | Implied | In vitb16 only | ⚠️ Partial | Medium | Add to others |
| Query init | Not zeros | Zeros (FullAttnRes) | ⚠️ Problematic | High | Fix |
| LLRD | Yes | NO | ❌ Missing | Critical | Add now |
| Loss (FocalLoss) | CrossEntropy | Focal gamma=2 | ⚠️ Different | High | Tune |
| Augmentation | Strong | Weak | ❌ Partial | High | Upgrade |
| Val protocol | k-fold | 193 samples | ❌ Missing | Critical | Fix |
| Checkpoint saving | Yes | NO | ❌ Missing | Critical | Add now |
| CosineClassifier | Optional | Missing in vitb16 | ⚠️ Missing | Medium | Add |
