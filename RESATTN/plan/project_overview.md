# Project Overview — ISIC2018 Skin Lesion Classification with Residual Attention

**Date**: 2026-05-09  
**Analyst**: AI Research Lab Assistant  
**Status**: Active Research — Target 85%+ Test Acc@1

---

## 1. Project Structure

```
ResTrans/
├── config.py                        # Global hyperparameters
├── train.py                         # Unified training script
├── test.py                          # Standalone test runner
├── requirements.txt                 # Dependencies
│
├── data/
│   ├── ISIC2018.py                  # Dataset class (CSV-based label loading)
│   └── ISIC2017.py                  # Unused (leftover)
│
├── model/
│   ├── basemodel.py                 # Baseline: ResNet152, ConvNeXt, ViT, SwinV2, DINOv2
│   ├── resattn.py                   # FullAttnRes: dense cross-block attention (DINOv2/ViT)
│   ├── block_resattn.py             # BlockAttnRes: block-sparse cross-attention (DINOv2)
│   ├── vitb16_resattn.py            # ViTB16_AttnRes: gated ResAttn on ViT-B/16
│   ├── conv_resattn.py              # ConvNeXt_AttnRes: ResAttn on ConvNeXt-Base
│   └── vit_moe.py                   # ViT_BlockMoE: ViT + per-block MoE aggregation
│
├── utils/
│   └── utils.py                     # Loss, augmentation, metrics, logger
│
├── logs/                            # Earlier experiment logs
└── logs_vitb16_resattn/             # Recent experiment logs (ViTB16 + ConvResAttn)
```

---

## 2. Dataset

| Split | Samples | Notes |
|-------|---------|-------|
| Train | 10,015 | Heavily imbalanced |
| Val   | **193** | ⚠️ Critically small |
| Test  | 1,512  | Official evaluation set |

### Class Distribution (ISIC2018 — 7 classes)

| Class | Name | Train N | Train % | Test N | Test % |
|-------|------|---------|---------|--------|--------|
| 0 | MEL (Melanoma) | 1113 | 11.11% | 171 | 11.31% |
| 1 | NV (Nevus) | 6705 | **66.95%** | 909 | 60.12% |
| 2 | BCC | 514 | 5.13% | 93 | 6.15% |
| 3 | AKIEC | 327 | 3.27% | 43 | 2.84% |
| 4 | BKL | 1099 | 10.97% | 217 | 14.35% |
| 5 | DF | **115** | **1.15%** | 44 | 2.91% |
| 6 | VASC | **142** | **1.42%** | 35 | 2.31% |

**Critical problem**: Val set has **only 1 sample of Class 5** and **3 of Class 6**. Validation metrics are completely unreliable for minority classes. This causes misleading model selection.

---

## 3. Architecture Overview

### 3.1 Baseline Models (`basemodel.py`)

Standard fine-tuning with:
- CosineClassifier head (normalized dot product × scale=30)
- Backbone frozen/partial, last few blocks fine-tuned
- Supports: ResNet152, ConvNeXt-B, ViT-B/16, SwinV2-B, DINOv2-ViT-B/14

### 3.2 FullAttnResClassifier (`resattn.py`)

**Core idea**: Each transformer block receives a weighted combination of ALL previous block outputs.

```
At block i:
  V = stack([x_0, x_1, ..., x_{i-1}, x_partial])  # [L, B, T, D]
  scores = einsum('d, lbtd -> lbt', query, norm(V))  # learnable query vector
  attn = softmax(scores, dim=0)  # across blocks
  h = einsum('lbt, lbtd -> btd', attn, V)
```

Problem: `query` initialized to zeros → all blocks equally weighted at init. No gating mechanism → no gradient preservation for pretrained weights.

### 3.3 BlockAttnResClassifier (`block_resattn.py`)

**Core idea**: Block-sparse version. Saves state every `block_size` layers.

**Critical Bug**: In `BlockAttnResViT.forward()`:
```python
final_res = BlockAttnRes(self.dim)  # ← NEW object created every forward!
x = self.final_res(blocks, partial)
```
`final_res` is instantiated inside `forward()` but `self.final_res` is used → the local variable is unused, `self.final_res` is used (which IS registered). But the local `final_res` has uninitialized weights that are never trained. This is a scope issue — luckily `self.final_res` is the real one, but the code is confusing.

**Another issue**: When `partial` becomes `None` at block boundary, the next layer gets `None` as hidden_states → will crash if blocks is empty.

### 3.4 ViTB16_AttnRes (`vitb16_resattn.py`)

**Core idea**: Gated ResAttn on ViT-B/16 with:
- Learned projection `Linear(D, 1)` instead of query vector
- Gated residual: `h = partial + gamma * (h_blend - partial)`, gamma init=0
- Zero-init of both `proj.weight` and `gamma` → preserves pretrained at init ✓

```
block_attn_res:
  V = stack(blocks + [partial])  # [N+1, B, T, D]
  logits = einsum('d, nbtd -> nbt', proj.weight, norm(V)) / sqrt(D)
  attn = softmax(logits, dim=0)
  out = einsum('nbt, nbtd -> btd', attn, V)
  return partial + gamma * (out - partial)
```

This is the most principled design. The `/sqrt(D)` temperature scaling and zero-init gating are the key innovations.

### 3.5 ConvNeXt_AttnRes (`conv_resattn.py`)

Extension of ResAttn to CNN. Intra-stage block attention + cross-stage feature aggregation via `CrossStageAttnRes`.

**Problem**: Train loss of 4.65 at epoch 1 (with FocalLoss gamma=2) → severe optimization instability.

### 3.6 ViT_BlockMoE (`vit_moe.py`)

Block-level aggregation using Dense MoE (Mixture of Experts). Each block of 4 layers → MoE → block classifier. Final fusion via global MoE. Auxiliary block-level loss at weight 0.3.

---

## 4. Training Pipeline

```
Dataset → WeightedRandomSampler → DataLoader
                ↓
    [per batch] Random r < 0.5:
      - r < 0.5 AND USE_MIXUP → Mixup (standard or class-aware)
      - r >= 0.5 AND USE_CUTMIX → CutMix (standard or class-aware)
                ↓
    Forward → FocalLoss(gamma=2) or CrossEntropy
                ↓
    AdamW(lr=1e-4, wd=1e-4) → CosineAnnealingLR(T_max=100)
```

### Key Hyperparameters

| Param | Value |
|-------|-------|
| Batch size | 32 |
| LR | 1e-4 |
| Weight decay | 1e-4 |
| Epochs | 100 |
| Scheduler | CosineAnnealing (eta_min=1e-6) |
| Loss | FocalLoss(gamma=2, alpha=None) |
| Mixup alpha | 0.4 |
| CutMix alpha | 1.0 |
| Imbalance | WeightedRandomSampler |

---

## 5. Evaluation Pipeline

- Metrics: Acc@1, Acc@5, Precision, Recall, F1 (macro), ROC-AUC (macro OvR)
- Eval every 10 epochs on test set
- **No checkpoint saving** — final model not persisted

---

## 6. Performance Summary (All Experiments)

| Model | Test Acc@1 | Macro F1 | ROC-AUC | Epochs | Notes |
|-------|-----------|----------|---------|--------|-------|
| resattn (DINOv2, no aug) | ~46% @Ep40 | ~0.39 | 0.848 | 100 | No pretrained backbone loaded! |
| block_resattn (DINOv2, cosine) | ~62% @Ep40 | ~0.49 | 0.899 | 100 | Slow convergence |
| vit_moe (DINOv2, cosine) | **83.3% @Ep20** | 0.776 | 0.958 | 2 only | Best early convergence |
| swinv2 (baseline) | 83.1% @Ep100 | 0.757 | 0.930 | 100 | Baseline strong |
| convnext (cosine, baseline) | **86.1%** | **0.785** | **0.946** | 100 | Best so far |
| vitb16_resattn (focal loss) | 75.3% @Ep20, **degrades** | 0.664 | 0.922 | 35 | Performance drops over time |
| conv_resattn (focal loss) | — | — | — | ~5 | Loss = 4.65, unstable |

---

## 7. Dependency Graph

```
config.py
    └── train.py
            ├── data/ISIC2018.py
            ├── utils/utils.py (FocalLoss, transforms, metrics, mixup/cutmix)
            └── model/
                    ├── basemodel.py (standalone)
                    ├── resattn.py (standalone)
                    ├── block_resattn.py (standalone)
                    ├── vitb16_resattn.py (standalone)
                    ├── conv_resattn.py (standalone)
                    └── vit_moe.py (standalone)
```

---

## 8. Identified Weak Points

### Critical (Performance Blockers)
1. **No checkpoint saving** — best model is never saved; final model may be suboptimal
2. **Val set is 193 samples** — model selection is completely unreliable; val metrics swing ±8% randomly
3. **vitb16_resattn degrades after Ep10** — test acc drops from 79.2% → 75.3% → 73.1%
4. **conv_resattn numerically unstable** — FocalLoss(gamma=2) + large model → loss=4.65 at epoch 1
5. **resattn/block_resattn use DINOv2 without loading pretrained weights** — training from scratch!

### High Priority (Architecture)
6. **FullAttnRes query initialized to zeros** — `nn.Parameter(torch.zeros(dim))` → all blocks weighted equally forever if loss landscape is flat near zero
7. **block_resattn: final_res created outside init but shadowed in forward** — confusing and potentially wrong aggregation
8. **CutMix modifies x in-place** (`x[:, :, bbx1:bbx2, bby1:bby2] = ...`) — corrupts the original batch tensor

### Medium Priority (Training)
9. **WeightedRandomSampler + Mixup/CutMix is redundant imbalance handling** — using both simultaneously may over-compensate
10. **No early stopping** — trains beyond peak performance (vitb16_resattn: best at Ep10, trained to Ep100+)
11. **No gradient clipping** — combined with FocalLoss can lead to gradient spikes
12. **Augmentation is very weak** — only RandomFlip, Rotation(10), light ColorJitter; no skin-specific augmentation
13. **cosine_classifier not used for vitb16_resattn** — inconsistent with best-performing baselines

### Low Priority (Engineering)
14. **No mixed precision (AMP)** — training 2× slower than necessary
15. **No experiment tracking** (no W&B, no TensorBoard integration active)
16. **Logger name collides** — uses fixed name "ISIC_Logger", multiple runs in same process share same logger
17. **Global variables in train.py** — `optimizer`, `criterion`, `scheduler` are module-level globals accessed inside `train_one_epoch`

---

## 9. Bottlenecks

**Primary bottleneck**: vitb16_resattn training instability and degradation over time.

Root causes (ranked):
1. Learning rate (1e-4) is likely too high for fine-tuning ViT-B/16 pretrained weights — causes catastrophic forgetting
2. No layer-wise LR decay — all parameters of ViT (patches, attention, MLP) trained at same LR as new ResAttn parameters
3. FocalLoss with gamma=2 combined with softmax outputs that are already well-calibrated for pretrained model → gradient explosion at early epochs
4. Val set noise makes it impossible to detect degradation and stop training
5. ResAttn head and ViT backbone need differential LR

**Secondary bottleneck**: Class imbalance for minority classes (DF, VASC) is not effectively handled — val set has 1 sample of DF which makes monitoring impossible.

---

## 10. Technical Debt

- [ ] `ISIC2017.py` is unused — dead code
- [ ] `resattn.py` contains two backbone paths (DINOv2 and ViT-B/16) but the ViT-B/16 path is never tested (uses `pretrained=False`)
- [ ] `train.py` uses global variables for optimizer/criterion — not thread-safe and hard to test
- [ ] No configuration validation (e.g., batch size must be > 1 for BN layers)
- [ ] `compute_model_complexity` wraps thop in try/except silently — FLOPs always =0 if thop fails
- [ ] No seed fixing beyond numpy/torch — `DataLoader` workers use different seeds each run

---

## 11. Missing Components

| Component | Impact | Priority |
|-----------|--------|----------|
| Model checkpointing (save best) | Critical | 🔴 |
| Early stopping | High | 🔴 |
| Layer-wise learning rate decay (LLRD) | High | 🔴 |
| Gradient clipping | High | 🟠 |
| Mixed precision (AMP) | Medium | 🟠 |
| Proper val/test stratification | High | 🔴 |
| Experiment reproducibility (seed fix) | Medium | 🟠 |
| TensorBoard / W&B integration | Low | 🟡 |
| Skin-specific augmentation (dermoscopy) | High | 🟠 |
| Confusion matrix logging | Medium | 🟠 |
