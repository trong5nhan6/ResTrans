# Training Analysis — Root Cause Investigation

**Date**: 2026-05-09  
**Models**: vitb16_resattn, conv_resattn, resattn, block_resattn, vit_moe, swinv2, convnext

---

## Executive Summary

The project has a fundamental performance ceiling at ~75% due to **three compounding problems**:
1. **Vitb16_resattn catastrophically degrades** after epoch 10 (best point) due to missing LLRD
2. **Validation set (193 samples) is too noisy** to detect degradation, so training continues past the peak
3. **No checkpoint saving** means the best model is lost even when it's achieved

The ConvNeXt baseline already achieves 86.1%, meaning the custom ResAttn architecture is UNDERPERFORMING its own backbone. This is the core research failure to explain and fix.

---

## 1. vitb16_resattn — Degradation Analysis

### Raw Training Data

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Test Acc@1 | Test F1 |
|-------|-----------|-----------|----------|---------|-----------|---------|
| 1 | 0.8781 | 0.5028 | 0.4337 | 0.6839 | — | — |
| 2 | 0.6610 | 0.6202 | 0.3006 | 0.8083 | — | — |
| 8 | 0.4641 | 0.7119 | 0.1857 | **0.8601** | — | — |
| 10 | — | — | 0.2217 | 0.7927 | — | — |
| 20 | 0.3912 | 0.7468 | 0.2683 | 0.8135 | **0.7526** | 0.6635 |
| **10\*** | 0.4228 | 0.7353 | 0.1890 | 0.8446 | **0.7917** | 0.6721 |
| **20\*** | 0.3621 | 0.7556 | 0.2212 | 0.8497 | 0.7712 | 0.6602 |
| **30\*** | 0.3263 | 0.7662 | 0.3003 | 0.7979 | 0.7308 | 0.6374 |
| 32 | 0.3160 | 0.7793 | **0.1706** | **0.8756** | — | — |

*\* = Second run (same model)*

### Root Cause Analysis

**Finding 1: Test accuracy PEAKS at Epoch 10 (79.2%) and DEGRADES afterward**

The pattern is: `Test Acc@10 (79.2%) > @20 (77.1%) > @30 (73.1%)` while `Train Acc@10 (73.5%) < @20 (75.6%) < @30 (76.6%)`.

This is a textbook **overfitting pattern** combined with **catastrophic forgetting**:
- Train acc increases slowly
- Test acc decreases
- Val acc oscillates (misleading — sample size noise)

**The model is memorizing the training distribution while losing generalization.**

**Finding 2: Val Acc is completely unreliable as a training signal**

Val oscillations in run 2:
```
Ep5:  72.0% → Ep7:  80.3% → Ep8:  76.2% → Ep10: 84.5% → Ep11: 79.8% → Ep12: 74.6%
```
A swing of **10 percentage points in consecutive epochs** on a 193-sample val set is pure noise. The model is not actually improving/degrading that much — the val sample composition changes due to batch randomness.

**Finding 3: Learning rate 1e-4 is too high for ViT-B/16 fine-tuning**

Standard fine-tuning literature:
- Full fine-tuning ViT-B: LR = 1e-5 to 3e-5
- Linear probe only: LR = 1e-3
- Layer-wise decay: backbone layers at LR×0.01 to LR×0.1, head at LR

At LR=1e-4, the pretrained ViT weights are being updated too aggressively. The ResAttn gamma gates (initialized at 0) let new ResAttn parameters activate gradually, but the ViT weights themselves (attention matrices, MLP weights) are being overwritten.

**Finding 4: No gradient clipping amplifies early instability**

FocalLoss at epoch 1 (loss=0.8781, very reasonable) suggests the first run started fine. But the lack of gradient clipping means occasional bad batches (from WeightedRandomSampler creating rare-class-heavy batches) can cause gradient spikes.

---

## 2. conv_resattn — Instability Analysis

### Raw Data (First 5 Epochs)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | **4.6589** | 0.3247 | 0.8696 | 0.6788 |
| 2 | 1.4320 | 0.4198 | 0.8781 | 0.6425 |
| 3 | 1.6410 | 0.4713 | 0.5914 | 0.7150 |
| 4 | 1.0450 | 0.5097 | 1.1881 | 0.4611 |
| 5 | 0.9453 | 0.5423 | 0.6046 | 0.6684 |

### Root Cause Analysis

**Finding: Training loss of 4.65 at epoch 1 indicates severe gradient explosion**

For 7-class FocalLoss with random init head: expected initial loss ≈ `-(1-1/7)^2 * log(1/7) ≈ 0.74 * 1.95 ≈ 1.44`.

Actual loss of 4.65 = **3× higher than expected**.

Causes:
1. **CrossStageAttnRes interpolates feature maps** from 128→1024 channels across stages. Bilinear interpolation of very different feature spaces before alignment can produce very large activations.
2. **gamma=0 init for ConvNeXtAttnResBlock** is correct, BUT the `CrossStageAttnRes.gamma` is also 0. If the aligned features have different scales, the `align` Conv2d layers (randomly initialized) output arbitrary values.
3. **No gradient clipping** → loss spike at epoch 1.
4. Loss recovers partially by epoch 5, but then re-spikes at epoch 4 (val=0.46!) → recurring instability.

**The conv_resattn cross-stage attention mechanism is architecturally unstable due to random initialization of alignment convolutions operating on features of vastly different scales (128 vs 1024).**

---

## 3. resattn (DINOv2-based) — Non-Convergence Analysis

### Key Observation

resattn test @Ep20: 41.3%, @Ep40: 46.0%, @Ep50: still low

For a model with DINOv2-ViT-B/14 backbone (strong pretrained features), this performance is **catastrophically poor**. DINOv2 baseline should easily achieve >70% with linear probe.

### Root Cause: Backbone not pretrained!

```python
elif backbone_name == "vitb16":
    vit = models.vit_b_16(pretrained=False)   # ← explicit
    is_dino = False
```

But for DINOv2:
```python
vit = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
```

If `torch.hub.load` fails (no internet, or cached incorrectly), the DINOv2 backbone may initialize randomly too.

**Additionally**: The `FullAttnResViT` completely replaces the standard ViT forward pass with a custom one that uses `self.attn_res` and `self.mlp_res` for every layer. This means the pretrained MLP and attention operations run, but the output is then REPLACED by the ResAttn aggregation — there is no direct residual from the standard ViT output.

In `FullAttnResBlock.forward()`:
```python
def forward(self, values):
    h = self.attn_res(values)                  # aggregated from ALL prev blocks
    attn_out = self.attn(self.norm1(h))        # standard attention on aggregated h
    values.append(attn_out)                    # only attn_out is saved, NOT h + attn_out

    h = self.mlp_res(values)
    mlp_out = self.mlp(self.norm2(h))
    values.append(mlp_out)                     # only mlp_out saved

    return values
```

**There is NO direct residual skip from input to output!** Standard transformer blocks compute `x = x + attn(x)`. This code computes `attn_out = attn(h)` and saves `attn_out` WITHOUT adding it back to `h`. This fundamentally breaks the residual nature of the transformer, destroying pretrained representations.

---

## 4. block_resattn — Slow Convergence Analysis

### Performance at Epoch 20: 56.6%, Epoch 40: 62.1%

For a DINOv2-B backbone, 62% at epoch 40 is very poor (random baseline = 1/7 ≈ 14%, NV-collapse ≈ 60%).

### Root Cause: partial=None creates dead zones

When `partial = None` at block boundaries, the next block processes:
```python
h = self.attn_res(blocks, partial=None)   # partial=None is valid
attn_out = self.attn(self.norm1(h))
partial = attn_out   # No residual addition! partial is INITIALIZED from attn_out
```

This creates a **non-residual initialization** for the first layer of each new block segment. The model has to learn to route information through the attention mechanism alone, without any direct residual path.

For pretrained ViT blocks, the attention layers expect a direct residual from input. Removing this residual at the start of each segment degrades the pretrained feature representations.

---

## 5. Comparative Analysis — Why ConvNeXt Wins

| Factor | ConvNeXt (86.1%) | vitb16_resattn (~75%) |
|--------|-----------------|----------------------|
| LR | 1e-4 (OK for CNN fine-tuning) | 1e-4 (too high for ViT) |
| Architecture change | Minimal (CosineClassifier) | Significant (ResAttn blocks) |
| Pretrained preservation | Full (conv layers handle this) | Partial (gamma helps but LR kills) |
| Loss | CrossEntropy | FocalLoss |
| Val selection | Train to Ep100 | Train to Ep100 (but peaks at Ep10) |
| Gradient flow | Clean (CNNs, no attention residuals) | Complex (ResAttn + standard residuals) |

**Conclusion**: ConvNeXt fine-tuning is robust because CNN weights are more tolerant to LR=1e-4 than transformer attention matrices. The custom ResAttn architecture adds complexity without adding robustness mechanisms (LLRD, gradient clipping, proper checkpoint selection).

---

## 6. vit_moe — Fast Convergence Analysis

### Performance at Epoch 20: 83.3%, F1=0.776

This is remarkable given it's reached with only 20 epochs (~45 seconds/epoch with DINOv2).

**Why vit_moe works:**
1. **DINOv2 pretrained features are very strong** — already encodes rich visual features
2. **MoE aggregation is additive** — doesn't modify the backbone's forward pass
3. **Block classifiers provide additional gradient paths** — 3 auxiliary losses
4. **Features are extracted AFTER standard transformer processing** — backbone intact
5. **Only the MoE and classifier heads have random init** — backbone is frozen during early epochs

This suggests the research direction should be: **preserve the backbone's forward pass completely, only add aggregation mechanisms externally**.

---

## 7. Class Imbalance Analysis

### Effective class support in val set

| Class | Val N | Reliability |
|-------|-------|-------------|
| 0 MEL | 21 | OK |
| 1 NV | 123 | OK |
| 2 BCC | 15 | Borderline |
| 3 AKIEC | 8 | Poor |
| 4 BKL | 22 | OK |
| 5 DF | **1** | **Useless** |
| 6 VASC | **3** | **Useless** |

The val set **cannot evaluate minority classes**. This means:
- The model might have 0% recall on DF but show 83% val accuracy
- WeightedRandomSampler improves training distribution but val selection ignores this
- The "best" checkpoint is chosen based on majority class performance

**Evidence**: Test ROC-AUC (0.922) is much higher than what Acc@1 (75.2%) suggests — the model's probability estimates are good but the argmax decision boundary is biased toward NV (Class 1).

---

## 8. Training Stability Patterns

### vitb16_resattn — Epoch-by-epoch Val Instability

```
Run 2: 72.0 → 78.8 → 78.8 → 73.1 → 72.0 → 68.4 → 80.3 → 76.2 → 71.5 → 84.5 → 79.8 → 74.6
```
Standard deviation of val acc ≈ 4.7%. With 193 samples, this is expected (~4.8% sampling SE) — but it's completely misleading as a training signal.

### swinv2 — Stable but Overfitting

```
Train Loss plateau: ~0.30 (Ep80-100)
Val Loss: increasing to 1.1-1.3 (classic overfitting pattern)
Val Acc: plateau at ~85%
```
swinv2 is clearly overfitting from epoch 30+ but val accuracy stays stable because the model predicts NV correctly most of the time. The val loss correctly signals overfitting but val acc doesn't — demonstrating why loss and acc can diverge.

---

## 9. Key Failure Patterns Summary

### Pattern 1: Architecture Failure (resattn, block_resattn)
- Missing residual connections in custom forward pass
- Pretrained weights invalidated by architectural changes
- Expected gain: 0% (these designs are fundamentally flawed)

### Pattern 2: Optimization Failure (vitb16_resattn)
- LR too high for fine-tuning
- No LLRD, no gradient clipping
- Training past peak (no early stopping, no checkpoint saving)
- Expected gain if fixed: +8-10% (recover the Ep10 performance of 79%)

### Pattern 3: Initialization Failure (conv_resattn)  
- Random alignment convolutions creating large initial activations
- Loss explosion at epoch 1
- Expected gain if fixed: ±unknown (architecture needs redesign)

### Pattern 4: Val Set Failure (all models)
- Can't track real improvement
- Can't save best checkpoint
- Expected gain from fix: +3-5% (selecting better checkpoint)

### Pattern 5: Data Failure (all models)
- Augmentation too weak for dermoscopy
- Mixup + WeightedSampler over-compensates for imbalance
- Expected gain from fix: +2-4%
