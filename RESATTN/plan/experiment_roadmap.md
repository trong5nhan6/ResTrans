# Experiment Roadmap — Prioritized Action Plan

**Date**: 2026-05-09  
**Target**: ≥85% Test Acc@1, ≥0.78 Macro F1  
**Baseline**: ConvNeXt 86.1% (already there), vitb16_resattn ~75%, need ResAttn to match ConvNeXt

---

## Phase 0 — Critical Infrastructure Fixes (Do FIRST, no exceptions)

These are not experiments — they are prerequisite fixes without which **all future experiments are invalid**.

| # | Fix | Why Critical | Files | Time |
|---|-----|-------------|-------|------|
| F1 | Add `save_best_model()` checkpoint saving | Without this, the best model is LOST | train.py | 1h |
| F2 | Add early stopping (patience=15) | vitb16_resattn peaks at Ep10, trained to Ep100 | train.py | 30min |
| F3 | Fix `FullAttnResBlock` — add residual connections | Missing `x = x + attn_out` destroys pretrained features | resattn.py | 1h |
| F4 | Fix `BlockAttnResLayer` — remove `partial = None` reset | Cuts gradient paths at block boundaries | block_resattn.py | 30min |
| F5 | Fix CutMix in-place modification bug | `x[:, :, ...] = x[index, ...]` corrupts batch | utils.py | 30min |
| F6 | Set random seed (torch, numpy, python) for reproducibility | Can't compare experiments without seed control | train.py | 15min |
| F7 | Add gradient clipping (`max_norm=1.0`) | Prevents gradient spikes with FocalLoss | train.py | 15min |

**Total estimated time**: 4-5 hours.

---

## Phase 1 — Quick Wins (High Impact, Low Compute)

**Target**: Push vitb16_resattn from 75% → 83-85%

### Experiment 1.1 — Layer-wise LR Decay (LLRD)
**Expected gain**: +6-10% Test Acc@1 for vitb16_resattn  
**Compute**: Same as current (~2 hours/run)

```
Config: LR_base=1e-4, LLRD_decay=0.8
Layer LR:
  - New ResAttn params: 1e-4
  - ViT encoder layer 12 (last): 1e-4 × 0.8^0 = 1.0e-4
  - ViT encoder layer 11: 1e-4 × 0.8^1 = 8.0e-5
  - ViT encoder layer 1 (first): 1e-4 × 0.8^11 = 8.6e-6
  - Patch embedding: 1e-5
```

**Ablation**: Run with decay=[0.65, 0.75, 0.80, 0.85] — 4 runs.

### Experiment 1.2 — Replace FocalLoss with CrossEntropy + Class Weights
**Expected gain**: +2-4% Test Acc@1 (via training stability)  
**Compute**: Zero additional cost

```
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

**Rationale**: FocalLoss(gamma=2) at epoch 1 with loss=4.65 shows instability. Class-weighted CE is more stable and well-calibrated.

**Ablation**: CE-noweight vs CE+weight vs FocalLoss(gamma=1) vs FocalLoss(gamma=2).

### Experiment 1.3 — CosineClassifier for vitb16_resattn
**Expected gain**: +1-2%  
**Compute**: Zero additional cost

```python
self.head = CosineClassifier(dim, num_classes, scale=30.0)
```

**Rationale**: ConvNeXt with CosineClassifier achieves 86.1% vs without it. The normalized classifier prevents weight norms from dominating minority-class predictions.

### Experiment 1.4 — Stronger Augmentation (albumentations)
**Expected gain**: +2-4%  
**Compute**: Slight increase in data loading time (~5-10%)

Replace `utils.get_transform()` with albumentations-based pipeline including:
- RandomResizedCrop(scale=0.7-1.0)
- ElasticTransform
- GridDistortion
- Stronger ColorJitter
- GaussNoise

### Experiment 1.5 — Test-Time Augmentation (TTA)
**Expected gain**: +1-2% (free at inference)  
**Compute**: ×4-8 inference cost, no training change

Apply 8 TTA variants at test time: original + 3 rotations + flips.

---

## Phase 2 — Medium-term Architecture Improvements

**Target**: Build a ResAttn model that outperforms ConvNeXt baseline (>87%)

### Experiment 2.1 — vitb16_resattn with LLRD + Warmup + Better Loss
**This is the consolidated "fixed" training run**

```
LR: 1e-4 (base), LLRD decay=0.75
Warmup: 5 epochs (linear)
Loss: CrossEntropy(weight=class_weights)
Scheduler: CosineAnnealing with warmup
Clip: max_norm=1.0
Checkpoint: save best val F1 (not val acc!)
```

**Expected**: ~83-86% Test Acc@1

### Experiment 2.2 — Multi-Scale Feature Extraction from ViT
**Idea**: Extract CLS token from layers 3, 6, 9, 12 and aggregate them

```python
layer_feats = []
for i, blk in enumerate(self.vit.encoder.layers):
    blocks, partial = blk(blocks, partial)
    if i in [2, 5, 8, 11]:  # every 3rd layer
        layer_feats.append(partial[:, 0])  # CLS token

# Aggregate
fused = self.aggregator(torch.stack(layer_feats, dim=1))  # [B, 4, D]
```

**Expected gain**: +2-3%  
**Compute**: Similar (just save intermediate CLS tokens)

### Experiment 2.3 — Fix FullAttnRes + DINOv2 with Proper Pretraining
**Fix the fundamental residual bug, retrain resattn with DINOv2**

```python
# Fixed FullAttnResBlock.forward():
def forward(self, values):
    h = self.attn_res(values)
    attn_out = self.attn(self.norm1(h))
    out = h + attn_out                    # ← ADDED: proper residual
    values.append(out)                    # ← save the full output, not just attn_out

    h2 = self.mlp_res(values)
    mlp_out = self.mlp(self.norm2(h2))
    out2 = h2 + mlp_out                  # ← ADDED: proper residual
    values.append(out2)
    return values
```

**Expected gain**: +20-30% over current resattn (should go from 46% → 70-80%)

### Experiment 2.4 — LDAM Loss for Minority Class Improvement
**Target**: Improve Macro F1 for DF (Class 5) and VASC (Class 6)

```python
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        # Compute per-class margin inversely proportional to n_i^{1/4}
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # ...
```

**Expected gain**: +3-5% Macro F1 for minority classes

### Experiment 2.5 — Decoupled Training (2-Stage)
**Stage 1**: Train backbone + ResAttn with full training (100 epochs)  
**Stage 2**: Freeze backbone, re-train head with LDAM loss (30 epochs) on balanced sampler

**Expected gain**: +3-5% Macro F1 without hurting overall accuracy

---

## Phase 3 — Long-term Research Directions

**Target**: Novel contributions toward publication quality

### Experiment 3.1 — Hierarchical ResAttn (Block + Cross-Stage)
**Design**: Apply ResAttn at two levels:
1. Intra-block (within each 4-layer block)
2. Cross-block (between block outputs)

The current vitb16_resattn is level 1 only. Adding cross-stage aggregation (like conv_resattn but for ViT) could add level 2.

**Expected gain**: +1-3% over vitb16_resattn  
**Research novelty**: Medium

### Experiment 3.2 — Asymmetric Dual Backbone (CNN + ViT Fusion)
**Design**: Run ConvNeXt and ViT-B in parallel, fuse features

```python
class DualBackboneFusion(nn.Module):
    def __init__(self):
        self.cnn = ConvNeXt_Base (pretrained)  # local texture expert
        self.vit = ViTB16_AttnRes (pretrained)  # global context expert
        self.fusion = CrossAttentionFusion(dim=1024+768)
```

**Rationale**: CNN excels at local texture (BCC, AKIEC — texture-based), ViT excels at global structure (MEL, NV — shape-based). Fusion should outperform either alone.  
**Expected gain**: +2-4% over best single model  
**Research novelty**: Medium-High  
**Compute**: 2× per forward pass

### Experiment 3.3 — Prototype-Based Classifier for Extreme Imbalance
**Paper**: "Few-Shot Learning with Prototypical Networks" (Snell et al., 2017)  
**Key idea**: Instead of a linear classifier, use class prototypes (mean of embeddings per class)

```python
# Prototype head
class PrototypeHead(nn.Module):
    def forward(self, x, prototypes):
        # x: [B, D], prototypes: [C, D]
        dists = torch.cdist(x, prototypes)  # [B, C]
        return -dists  # logits = negative distance
```

**Why relevant**: For DF (115 samples) and VASC (142 samples), prototype-based classification is more robust than learning a linear weight from few examples.  
**Expected gain**: +5-10% Macro F1 on minority classes

### Experiment 3.4 — Self-Supervised Pre-training on ISIC Data
**Key idea**: Mask ~75% of dermoscopy patches (MAE-style), train ViT-B to reconstruct

```bash
# Step 1: MAE pretraining
python pretrain_mae.py --data ./datasets/ISIC2018/train --epochs 200 --mask_ratio 0.75

# Step 2: Fine-tune with LLRD
python train.py --model vitb16_resattn --pretrain ./checkpoints/mae_isic.pth
```

**Expected gain**: +3-5% over ImageNet-pretrained ViT-B  
**Compute**: High (200 epoch pretraining)  
**Research novelty**: High (domain-specific MAE for skin)

### Experiment 3.5 — Attention Map Analysis + Regularization
**Idea**: Monitor ResAttn attention weights across training. If attention collapses (one block getting all weight), add entropy regularization:

```python
# Regularization: encourage uniform attention over blocks
attn_entropy = -(attn * attn.log()).sum(dim=0).mean()
loss = main_loss - 0.01 * attn_entropy  # maximize entropy = more uniform attention
```

**Why relevant**: The gamma_attn initialized at 0 means all blocks are equally weighted at start. As training progresses, if one block collapses to weight=1, the model is learning nothing from ResAttn.  
**Research novelty**: Medium-High (attention collapse analysis in ResAttn is novel)

---

## Priority Matrix

### Quick Wins (Phase 0+1): Implement First, High ROI

| Priority | Experiment | Why | Expected Gain | Cost | Risk | Dependencies |
|----------|-----------|-----|---------------|------|------|--------------|
| P0 | F1: Save best checkpoint | Lost best model | recovery | None | None | - |
| P0 | F2: Early stopping | Trains past peak | recovery | None | None | F1 |
| P0 | F3: Fix FullAttnRes residual | Architecture bug | +20-25% for resattn | None | Low | - |
| P0 | F4: Fix block_resattn reset | Architecture bug | +10-15% for block_resattn | None | Low | - |
| P0 | F5: Fix CutMix in-place | Data corruption bug | +1-2% | None | None | - |
| P0 | F6: Seed fixing | Reproducibility | baseline quality | None | None | - |
| P0 | F7: Gradient clipping | Instability | stability | None | None | - |
| P1 | 1.1: LLRD | Stops catastrophic forgetting | +6-10% | None | Low | F1,F7 |
| P1 | 1.2: Replace FocalLoss | Training stability | +2-4% | None | Low | - |
| P1 | 1.3: CosineClassifier | Better head | +1-2% | None | Low | - |
| P1 | 1.4: Albumentations | Better augment | +2-4% | Low | Low | - |
| P1 | 1.5: TTA | Free accuracy boost | +1-2% | None | None | F1 |

### Medium-term (Phase 2): After Phase 1 Complete

| Priority | Experiment | Why | Expected Gain | Cost | Risk | Dependencies |
|----------|-----------|-----|---------------|------|------|--------------|
| P2 | 2.1: Consolidated fixed vitb16 run | Validate all fixes together | 83-86% | Medium | Low | Phase 0+1 |
| P2 | 2.2: Multi-scale CLS aggregation | Better features | +2-3% | Low | Low | 2.1 |
| P2 | 2.3: Fixed resattn + DINOv2 | Test architecture fix | +20-30% vs broken | Low | Low | F3 |
| P2 | 2.4: LDAM loss | Minority class F1 | +3-5% F1 | None | Low | 2.1 |
| P2 | 2.5: Decoupled training | Better imbalance handling | +3-5% F1 | Low | Low | 2.1 |

### Long-term Research (Phase 3): Novel Contributions

| Priority | Experiment | Why | Expected Gain | Cost | Risk | Dependencies |
|----------|-----------|-----|---------------|------|------|--------------|
| P3 | 3.1: Hierarchical ResAttn | Novel architecture | +1-3% | Medium | Medium | Phase 2 |
| P3 | 3.2: Dual backbone fusion | CNN+ViT synergy | +2-4% | High | Medium | Phase 2 |
| P3 | 3.3: Prototype classifier | Minority class improvement | +5-10% F1 | Low | Low | Phase 2 |
| P3 | 3.4: MAE pretraining on ISIC | Domain adaptation | +3-5% | High | Low | Phase 2 |
| P3 | 3.5: Attention collapse analysis | Research novelty | +1-2% + insights | Low | Low | Phase 2 |

---

## Expected Progress Timeline

```
Week 1: Phase 0 (infrastructure fixes) → baseline is now reliable
Week 1-2: Phase 1.1-1.3 (LLRD, loss, head) → vitb16_resattn 83-85%
Week 2-3: Phase 1.4-1.5 + 2.1 → consolidated best model 85-87%
Week 3-4: Phase 2.2-2.5 → push toward 88% Macro F1 improvement
Month 2: Phase 3 → research-level contributions
```

---

## Ablation Study Plan

For each major change, run these ablations to isolate impact:

### LLRD Ablation
```
A: LR=1e-4 (uniform)     ← current
B: LLRD decay=0.65
C: LLRD decay=0.75       ← recommended
D: LLRD decay=0.85
E: Freeze backbone entirely + head only
```

### Loss Function Ablation
```
A: FocalLoss(gamma=2, alpha=None)  ← current
B: CrossEntropy (no weight)
C: CrossEntropy + class weights
D: FocalLoss(gamma=1.0, alpha=class_weights)
E: LDAM loss
F: Balanced Softmax
```

### Augmentation Ablation
```
A: Current (minimal)
B: +RandResizedCrop
C: B + ElasticTransform
D: C + stronger ColorJitter
E: Full dermoscopy pipeline
```
