# Research Survey — Related Work & SOTA Techniques

**Date**: 2026-05-09  
**Task**: Skin lesion classification, ISIC2018, 7-class  
**Relevant Domains**: Medical imaging, ViT fine-tuning, class imbalance, hybrid CNN-Transformer

---

## 1. Foundational Baselines (Understanding Why ConvNeXt Wins)

### 1.1 ConvNeXt-B (86.1% in this project)
**Paper**: "A ConvNet for the 2020s" (Liu et al., 2022)  
**Key ideas**: Modernized ResNet with depthwise conv, GELU, LayerNorm, larger kernels  
**Why relevant**: Already the best model in this project. Understanding what makes it work guides architecture design.  
**Key insight**: Depthwise convolutions efficiently capture local texture — crucial for skin lesion texture patterns.

### 1.2 SwinV2-B (83.1% in this project)
**Paper**: "Swin Transformer V2" (Liu et al., 2022)  
**Why it lags ConvNeXt**: Shifted window attention doesn't naturally capture global patterns in dermatology. Small window size (e.g., 8×8) may miss macro-level lesion context.

---

## 2. Fine-Tuning Strategies for ViT on Medical Imaging

### 2.1 Layer-wise LR Decay (LLRD)
**Reference**: Beit, DINOv2 fine-tuning recipes  
**Key idea**: `lr_layer_i = base_lr * decay^(num_layers - i)`, typical decay=0.65-0.85  
**Why relevant**: vitb16_resattn degrades because backbone LR=1e-4 is catastrophic. LLRD would set early ViT layers to LR ≈ 1e-6.  
**Expected impact**: +5-8% Test Acc@1 for vitb16_resattn  
**Compute cost**: None (optimizer configuration only)  
**Risk**: Low — well-established technique

### 2.2 MAE Fine-tuning Protocol (He et al., 2022)
**Reference**: "Masked Autoencoders Are Scalable Vision Learners"  
**Key recipe**: 
- LR = 1e-3 for linear probe, 1e-5 for full fine-tune
- Warmup 5-10 epochs
- Label smoothing 0.1
- Layer decay 0.65-0.75
**Why relevant**: MAE ViT-B/16 on dermoscopy achieves SOTA in several benchmarks  
**Expected impact**: Using MAE-pretrained ViT-B instead of ImageNet supervision could give +3-5%

### 2.3 EfficientNet / EfficientViT for Medical Imaging
**Reference**: "EfficientViT: Multi-Scale Linear Attention" (Cai et al., 2023)  
**Why relevant**: Efficient attention that captures multi-scale features — crucial for lesion at different magnifications  
**Compute cost**: Low-Medium  
**Risk**: Medium (requires swapping backbone)

---

## 3. ISIC2018 SOTA Methods (Direct Comparison)

### 3.1 Published Benchmarks
- **ISIC2018 official challenge winner**: ~85-88% balanced accuracy
- **EfficientNet-B7 fine-tuned**: ~85% accuracy
- **ResNet50 + attention**: ~82% accuracy  
- **DenseNet201**: ~83% accuracy
- **TransFuse (CNN + ViT hybrid)**: ~86% accuracy

### 3.2 SkinCon + CLIP-based methods (2023)
**Key idea**: Using vision-language alignment on skin lesion terminology  
**Expected impact**: Strong zero-shot baseline, fine-tuning could push >87%  
**Risk**: Requires CLIP model download

### 3.3 HAM10000 Augmentation Strategy (Cassidy et al., 2022)
**Reference**: "Analysis of the ISIC Image Datasets"  
**Key findings**:
- Hair removal augmentation: +1-2%
- Microscope circle masking: +0.5-1%
- Lesion-centric cropping: +1-3%
**Why relevant**: Directly applicable to ISIC2018 (same imaging conditions)

---

## 4. Class Imbalance Techniques

### 4.1 LDAM Loss (Label-Distribution-Aware Margin)
**Paper**: "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (Cao et al., 2019)  
**Key idea**: Adds class-dependent margin to the decision boundary, inversely proportional to class frequency  
**Formula**: `Loss_i = max(logit_j - logit_i + Δ_i, 0)` where `Δ_i ∝ n_i^{-1/4}`  
**Why relevant**: ISIC2018 has extreme imbalance (66:1 ratio). LDAM specifically designed for this.  
**Expected impact**: +2-4% Macro F1 (mainly improves minority class recall)  
**Compute cost**: Negligible  
**Risk**: Low — standard technique

### 4.2 Balanced Softmax Loss
**Paper**: "Balanced Meta-Softmax for Long-Tailed Visual Recognition" (Ren et al., 2020)  
**Key idea**: Modifies softmax to account for class frequency: `p_i = exp(f_i - log(n_i)) / Σ exp(f_j - log(n_j))`  
**Why relevant**: Mathematically correct correction for biased sampling. Works well with WeightedRandomSampler.  
**Compute cost**: Negligible  
**Risk**: Low

### 4.3 Decoupled Training (Two-Stage)
**Paper**: "Decoupled Representation and Classifier for Long-Tailed Recognition" (Kang et al., 2020)  
**Stage 1**: Train backbone with standard sampling (let ConvNeXt learn features)  
**Stage 2**: Freeze backbone, re-train only classifier with balanced sampling + LDAM  
**Why relevant**: Prevents the backbone from being distorted by imbalance-correction techniques  
**Expected impact**: +2-4% Macro F1  
**Compute cost**: Low (stage 2 is fast)  
**Risk**: Low

### 4.4 Meta-Weight-Net
**Paper**: "Meta-Weight-Net" (Shu et al., 2019)  
**Key idea**: Learn sample weights via meta-learning on clean validation set  
**Risk**: High — requires unbiased meta-validation set (our val set is too noisy)

---

## 5. Attention Mechanism Improvements

### 5.1 LayerScale (CaiT)
**Paper**: "Going Deeper with Image Transformers" (Touvron et al., 2021)  
**Key idea**: `output = x + alpha * sublayer(x)` where alpha is a per-channel learnable scalar, initialized to small value (1e-5 to 0.1)  
**Why relevant**: Similar to the gamma gating in vitb16_resattn, but more fine-grained (per-channel, not scalar)  
**Expected impact**: More stable fine-tuning, especially for deep ViT models  
**Compute cost**: Negligible  
**Risk**: Low — well-validated

### 5.2 Deformable Attention
**Paper**: "DAT: Deformable Attention Transformer" (Xia et al., 2022)  
**Key idea**: Learns to attend to sparse, deformable locations rather than fixed window/global tokens  
**Why relevant**: Skin lesions have irregular shapes — deformable attention can focus on lesion boundaries more naturally  
**Compute cost**: Medium  
**Risk**: Medium — requires architectural changes

### 5.3 Efficient Self-Attention (Linear Attention)
**Paper**: "FastViT", "EfficientViT" variants  
**Key idea**: Replace O(N²) attention with O(N) linear approximation  
**Why relevant**: At 224×224 with 16×16 patches, ViT-B has N=196 tokens. Linear attention enables larger input resolution (e.g., 384×384) which helps for skin lesions.  
**Expected impact**: Enabling 384×384 input → +2-4%  
**Compute cost**: Medium (need to retrain)

---

## 6. Multi-Scale Feature Extraction

### 6.1 Feature Pyramid Network (FPN) for Classification
**Reference**: RetinaNet, PVT  
**Key idea**: Aggregate features from multiple ViT layers with different receptive fields  
**Why relevant**: Skin lesions need both local texture (BCC, AKIEC) and global structure (MEL, NV). Multi-scale fusion handles this.  
**Implementation**: Use ViT's intermediate features at layers 3, 6, 9, 12 → FPN → classif.  
**Expected impact**: +2-3%  
**Risk**: Low

### 6.2 Squeeze-and-Excitation (SE) Blocks
**Paper**: SENet (Hu et al., 2018)  
**Key idea**: Channel-wise recalibration based on global context  
**Why relevant**: In ConvNeXt pipeline, adding SE blocks after each stage can improve channel selection  
**Compute cost**: Low  
**Risk**: Low

---

## 7. Test-Time Augmentation (TTA)

### Standard TTA
**Augmentations**: horizontal/vertical flip, 90° rotations, scale variation  
**Expected impact**: +1-2% reliable  
**Compute cost**: Low (×4-8 inference passes)  
**Risk**: None — only at test time  
**Note**: Dermoscopy images are rotation-invariant (no canonical orientation)

### Advanced TTA with Ensemble
Combine TTA + model ensemble (ConvNeXt + ViTB16_ResAttn predictions)  
Expected: +1-3%

---

## 8. Knowledge Distillation

### 8.1 Response-Based Distillation
**Teacher**: ConvNeXt (86.1%), DINOv2 (strong features)  
**Student**: vitb16_resattn or smaller model  
**Key idea**: Train student to match teacher's soft probability outputs  
**Expected impact**: +2-5% for student model (smaller gap from ConvNeXt)  
**Compute cost**: Medium (need teacher inference during training)  
**Risk**: Low

### 8.2 Feature-Based Distillation (DINOv2 → ViT-B/16)
**Key idea**: Force ViT-B/16 features to match DINOv2 features at intermediate layers  
**Why relevant**: DINOv2 vit_moe achieves 83%+ at epoch 20. Distilling DINOv2 knowledge into ViTB16_AttnRes could improve it significantly  
**Risk**: Medium

---

## 9. Self-Supervised Pretraining on Dermoscopy

### 9.1 MAE on ISIC2018
**Key idea**: Pretrain ViT-B/16 with MAE on all ISIC images (train + unlabeled) before supervised fine-tuning  
**Expected impact**: +3-5% (domain-specific pretraining > ImageNet pretraining for medical images)  
**Compute cost**: High (MAE pretraining ~200 epochs)  
**Risk**: Low (established technique)

### 9.2 MoCo-v3 / SimCLR on ISIC
**Key idea**: Contrastive pretraining on ISIC dermoscopy images  
**Expected impact**: +2-4%  
**Compute cost**: High  
**Risk**: Low

### 9.3 DINO Self-Distillation on ISIC
**Key idea**: Fine-tune DINOv2 teacher on ISIC domain with self-distillation  
**Expected impact**: +1-3% over standard DINOv2  
**Risk**: Medium (complex setup)

---

## 10. Dermoscopy-Specific Techniques

### 10.1 Hair Artifact Removal/Augmentation
**Paper**: "DullRazor" and hair augmentation methods  
**Key idea**: Augment training data with synthetic hair artifacts (or remove them during preprocessing)  
**Why relevant**: ISIC images often have hair occlusion which confuses classifiers  
**Expected impact**: +0.5-2%  
**Risk**: Low

### 10.2 Lesion-Aware Attention / Saliency Maps
**Key idea**: Use segmentation masks (available in ISIC2018 as optional) to guide attention  
**Expected impact**: +2-4% if masks are available  
**Risk**: Medium (requires segmentation dataset alignment)

### 10.3 Cross-Dataset Pretraining
**Key idea**: Pretrain on ISIC2016 + ISIC2017 + HAM10000 before fine-tuning on ISIC2018  
**Expected impact**: +2-5% (more dermoscopy-specific features)  
**Compute cost**: Medium  
**Risk**: Low (data leakage concern — need to check class overlap)

### 10.4 Dermoscopy-Specific Augmentation (Albumentations)
```python
import albumentations as A

train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
    A.ElasticTransform(alpha=1, sigma=50, p=0.3),
    A.GridDistortion(distort_limit=0.3, p=0.3),
    A.Sharpen(alpha=(0.2, 0.5), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),  # replaces CutOut
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```
**Expected impact**: +2-4%  
**Risk**: Low

---

## 11. Ensemble Strategies

### 11.1 Simple Model Ensemble
Models to combine: ConvNeXt (86.1%) + SwinV2 (83.1%) + ViTB16_AttnRes (best checkpoint)  
**Method**: Average softmax probabilities  
**Expected impact**: +1-3%  
**Risk**: None

### 11.2 Snapshot Ensemble
**Key idea**: Save checkpoints at different cosine LR minima → ensemble them  
**Expected impact**: +1-2% with no additional training cost  
**Risk**: Low (requires checkpoint saving first)

---

## 12. Summary Table — Priority Ranking

| Technique | Impact | Cost | Risk | Priority |
|-----------|--------|------|------|----------|
| LLRD for vitb16_resattn | +8-10% | None | Low | 🔴 P1 |
| Checkpoint saving + early stopping | +3-5% | None | None | 🔴 P1 |
| Dermoscopy augmentation (albumentations) | +2-4% | Low | Low | 🔴 P1 |
| Fix resattn forward (add residuals) | +15-20% | Low | Low | 🔴 P1 |
| Fix block_resattn (no partial reset) | +5-10% | Low | Low | 🔴 P1 |
| LDAM / Balanced Softmax | +2-4% F1 | None | Low | 🟠 P2 |
| TTA at inference | +1-2% | None | None | 🟠 P2 |
| CosineClassifier for vitb16_resattn | +1-2% | None | Low | 🟠 P2 |
| Multi-scale feature extraction | +2-3% | Medium | Medium | 🟠 P2 |
| Gradient clipping | stability | None | None | 🟠 P2 |
| LayerScale | stability | None | Low | 🟠 P2 |
| Decoupled training | +2-4% F1 | Low | Low | 🟡 P3 |
| Knowledge distillation | +2-5% | Medium | Low | 🟡 P3 |
| MAE pretraining on ISIC | +3-5% | High | Low | 🟡 P3 |
| Deformable attention | +1-3% | Medium | Medium | 🟡 P3 |
| Self-supervised pretraining | +2-4% | High | Low | ⚪ P4 |
| Larger resolution (384px) | +2-4% | High | Medium | ⚪ P4 |
