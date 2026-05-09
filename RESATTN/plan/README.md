# Research Plan — ISIC2018 ResAttn Project

**Created**: 2026-05-09  
**Analyst**: AI Research Lab Assistant  
**Goal**: Push ResAttn models from 75% → 85%+ Test Acc@1

---

## Critical Summary (Read This First)

### Why the models underperform:

1. **`resattn.py` has a critical architectural bug** — missing residual connections (`h + attn_out`) destroy pretrained features. Fix this FIRST. Expected gain: +25% absolute.

2. **`vitb16_resattn` catastrophically forgets** because LR=1e-4 is 10× too high for ViT fine-tuning. Layer-wise LR decay will fix this. Expected gain: +8-10%.

3. **No checkpoint saving** — the best model (vitb16_resattn at epoch 10: 79.2%) is permanently lost. Every experiment discards its best result.

4. **Val set has 193 samples (1 sample of Class 5)** — validation metrics are completely unreliable. Model selection is random.

5. **CutMix modifies input tensor in-place** — data corruption bug, subtle but real.

### What already works:
- ConvNeXt baseline: **86.1%** — the target is achievable
- vit_moe (DINOv2): **83.3% in 20 epochs** — DINOv2 features are strong
- ViTB16_AttnRes design (gated gamma, zero-init, LLRD-ready) — architecture is sound

---

## Plan Documents Index

### Analysis (Read these to understand the problem)

| File | Purpose |
|------|---------|
| `project_overview.md` | Full project structure, architecture overview, identified weak points |
| `paper_alignment_analysis.md` | Implementation vs paper comparison, what's wrong vs what's correct |
| `training_analysis.md` | Root cause analysis of all training failures, failure pattern classification |
| `research_survey.md` | SOTA techniques, relevant papers, what to try and why |
| `code_quality_review.md` | Code issues, technical debt, quick wins |
| `research_value_analysis.md` | Publishability, novelty, clinical value, competition potential |

### Planning (Consult these before implementing)

| File | Purpose |
|------|---------|
| `experiment_roadmap.md` | Prioritized experiments, timeline, ablation plans |

### Improvement Ideas (One per technique)

| File | Priority | Expected Gain | Complexity |
|------|----------|--------------|-----------|
| `improvement_ideas/idea_001_LLRD_layer_wise_lr_decay.md` | 🔴 P0 | +8-10% | Low |
| `improvement_ideas/idea_002_fix_resattn_residual_connections.md` | 🔴 P0 | +25-30% | Low |
| `improvement_ideas/idea_003_checkpoint_saving_early_stopping.md` | 🔴 P0 | +3-5% | Low |
| `improvement_ideas/idea_004_dermoscopy_augmentation.md` | 🟠 P1 | +2-4% | Low |
| `improvement_ideas/idea_005_LDAM_balanced_loss.md` | 🟠 P2 | +3-5% F1 | Low |
| `improvement_ideas/idea_006_multiscale_cls_aggregation.md` | 🟠 P2 | +2-3% | Medium |
| `improvement_ideas/idea_007_dual_backbone_cnn_vit_fusion.md` | 🟡 P3 | +2-5% | High |
| `improvement_ideas/idea_008_TTA_test_time_augmentation.md` | 🟠 P1 | +1-2% | Low |

### Implementation Tasks (Actionable — implement these)

| File | Status | Dependency |
|------|--------|-----------|
| `tasks/task_001_fix_critical_bugs.md` | ⚡ READY | None — DO FIRST |
| `tasks/task_002_vitb16_resattn_with_LLRD.md` | READY after Task 001 | Task 001 |
| `tasks/task_003_fix_resattn_retrain_dinov2.md` | READY after Task 001 | Task 001 |

---

## Execution Order

```
TODAY → Task 001 (bug fixes, ~4 hours)
         ↓
Day 2  → Task 002 (LLRD training, ~6 hours)
       → Task 003 (fixed resattn, ~6 hours)
         ↓
Day 3-4 → idea_004 (augmentation upgrade)
        → idea_008 (TTA at inference)
         ↓
Week 2  → idea_005 (LDAM loss)
        → idea_006 (multi-scale CLS)
         ↓
Month 2 → idea_007 (dual backbone)
```

---

## Performance Trajectory (Expected)

| Milestone | Test Acc@1 | Macro F1 | When |
|-----------|-----------|----------|------|
| Current (broken) | 75% | 0.664 | Now |
| After bug fixes | 78-80% | 0.68 | Week 1 |
| + LLRD | 83-86% | 0.72 | Week 1-2 |
| + Augmentation + TTA | 85-88% | 0.75 | Week 2-3 |
| + LDAM + multi-scale | 87-89% | 0.78 | Week 3-4 |
| + Dual backbone | 88-91% | 0.82 | Month 2 |

---

## Key Contacts / References

- **ISIC 2018 Challenge**: https://challenge.isic-archive.com/
- **DINOv2 Hub**: facebookresearch/dinov2
- **ViT-B/16 (IMAGENET1K_V1)**: torchvision.models.vit_b_16
- **ConvNeXt-B**: torchvision.models.convnext_base
- **LDAM Loss Paper**: Cao et al., NeurIPS 2019 (arXiv:1906.07413)
- **LayerScale/CaiT**: Touvron et al., ICCV 2021 (arXiv:2103.17239)
- **LLRD Reference**: DINOv2 fine-tuning guide, BEiT paper

---

*All suggestions in this plan are grounded in the actual code, actual logs, and actual performance numbers from this project. No generic advice.*
