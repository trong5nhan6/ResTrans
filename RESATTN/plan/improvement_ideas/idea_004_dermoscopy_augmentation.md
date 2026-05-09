# Idea 004 — Dermoscopy-Specific Data Augmentation

**Priority**: P1 (High — easy win, validated in literature)  
**Complexity**: Low  
**Compute Cost**: Low (~5-10% slower data loading)

---

## Motivation

Current augmentation pipeline is designed for generic ImageNet training:
```python
T.RandomHorizontalFlip()
T.RandomVerticalFlip()
T.RandomRotation(10)
T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)  # very weak
```

Dermoscopy images have specific characteristics that this pipeline ignores:
1. **No canonical orientation**: lesions can appear at any angle → need full 360° rotation
2. **Device-specific color artifacts**: different dermoscopes produce different color casts
3. **Hair artifacts**: synthetic hair occlusion as augmentation is beneficial
4. **Elastic deformation**: skin deforms, lesions stretch → elastic augmentation mimics this
5. **Lesion-centric crops**: the lesion might be centered or off-center in ISIC images

The ConvNeXt baseline reaches 86.1% with current augmentation. Improving augmentation is likely to push it further and is critical for the ResAttn models to generalize better.

---

## Current Problem

The weak ColorJitter (0.1 brightness/contrast) does not account for:
- Dermoscopy images from different devices (color temperature varies ±20-30%)
- Image acquisition artifacts (vignetting, lens distortion)
- Post-processing differences between clinical sites

The small rotation (10°) misses the key property that dermoscopy is rotationally symmetric — the diagnostic features (asymmetry, irregular borders) are rotation-invariant by definition.

---

## Hypothesis

Replacing the current pipeline with dermoscopy-specific augmentation will:
1. Improve model robustness to imaging variations
2. Reduce overfitting (more diverse training samples)
3. Particularly help minority classes (DF, VASC) by creating more effective training samples
4. Expected: +2-4% Test Acc@1, +3-5% Macro F1

---

## Expected Improvement

Literature evidence:
- Codella et al. (ISIC analysis): +2-3% with elastic/distortion augmentation
- HAM10000 augmentation study: +1.5-2% with microscope augmentation
- General medical imaging: +3-5% with domain-specific strong augmentation vs ImageNet-default

---

## Risk

Low. Augmentation can only improve generalization. The risk is:
- **Over-augmentation**: If augmentation is too strong, it can make the training problem too hard (e.g., rotating 90° + elastic transform + color jitter may destroy diagnostic features). Mitigated by conservative probability values.
- **Train-test mismatch**: Test set uses no augmentation. If train augmentation introduces artifacts not present in test, generalization can drop.

---

## Implementation Plan

### Step 1: Install albumentations

```bash
pip install albumentations --break-system-packages
```

### Step 2: Create new transform in `utils/utils.py`

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_dermoscopy_transform(is_train=True, image_size=224):
    """
    Dermoscopy-specific augmentation pipeline.
    
    Design principles:
    - High probability for geometric augmentations (lesions are rotation-invariant)
    - Moderate probability for color augmentations (device variation is common)
    - Low probability for destructive augmentations (noise, dropout)
    - Always normalize with ImageNet stats (using pretrained backbone)
    """
    if is_train:
        return A.Compose([
            # ===== Geometric augmentations (HIGH priority for dermoscopy) =====
            A.RandomResizedCrop(
                height=image_size, width=image_size,
                scale=(0.7, 1.0),       # 70-100% of original area
                ratio=(0.85, 1.15),     # slight aspect ratio variation
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),    # 90°, 180°, 270° rotations (dermoscopy-valid)
            A.Transpose(p=0.3),         # diagonal flip
            
            # ===== Elastic deformations (skin surface deformation) =====
            A.OneOf([
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50,
                    border_mode=cv2.BORDER_REFLECT,
                    p=1.0
                ),
                A.GridDistortion(
                    num_steps=5, distort_limit=0.3,
                    border_mode=cv2.BORDER_REFLECT,
                    p=1.0
                ),
            ], p=0.3),
            
            # ===== Color augmentations (device variation) =====
            A.ColorJitter(
                brightness=0.3,     # ±30% (was 0.1)
                contrast=0.3,
                saturation=0.3,
                hue=0.05,           # small hue shift
                p=0.8
            ),
            
            # ===== Medical imaging specific =====
            A.OneOf([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),  # contrast enhance
                A.Sharpen(alpha=(0.2, 0.4), lightness=(0.8, 1.2), p=1.0),
                A.UnsharpMask(blur_limit=(3, 5), sigma_limit=0.5, p=1.0),
            ], p=0.3),
            
            # ===== Noise and artifacts =====
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.2),
            
            # ===== Occlusion / Regularization =====
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),  # replaces Cutout
            
            # ===== Normalization (always last) =====
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    else:
        # Validation/Test: minimal transforms
        return A.Compose([
            A.Resize(image_size, image_size),
            A.CenterCrop(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
```

### Step 3: Update ISIC2018 Dataset to use albumentations

```python
# In data/ISIC2018.py
class ISIC2018(Dataset):
    def __init__(self, root_dir, transform=None, use_albumentations=False):
        # ... existing code ...
        self.use_albumentations = use_albumentations
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["image"]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        
        if self.use_albumentations:
            # Load as numpy array for albumentations
            import cv2
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
        else:
            # Standard PIL loading
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        
        label = self.get_label(row)
        return image, label
```

### Step 4: Hair Augmentation (Optional, High Impact)

```python
class SyntheticHairAugmentation(A.ImageOnlyTransform):
    """
    Adds synthetic hair strands to dermoscopy images.
    Based on: "DullRazor" artifact analysis.
    """
    def __init__(self, num_hairs=(0, 10), hair_width=(1, 3), 
                 hair_color=(0, 0, 0), p=0.3):
        super().__init__(p=p)
        self.num_hairs = num_hairs
        self.hair_width = hair_width
        self.hair_color = hair_color
    
    def apply(self, img, **params):
        import cv2
        import random
        
        result = img.copy()
        H, W = img.shape[:2]
        
        n_hairs = random.randint(*self.num_hairs)
        for _ in range(n_hairs):
            # Random hair strand as bezier curve
            x1, y1 = random.randint(0, W), random.randint(0, H)
            x2, y2 = random.randint(0, W), random.randint(0, H)
            width = random.randint(*self.hair_width)
            
            cv2.line(result, (x1, y1), (x2, y2), 
                     self.hair_color, width, lineType=cv2.LINE_AA)
        
        return result
    
    def get_transform_init_args_names(self):
        return ('num_hairs', 'hair_width', 'hair_color')
```

---

## Files Potentially Affected

- `utils/utils.py` — replace `get_transform()` with `get_dermoscopy_transform()`
- `data/ISIC2018.py` — add albumentations support
- `config.py` — add `USE_ALBUMENTATIONS = True`
- `requirements.txt` — add `albumentations>=1.3.0`

---

## Ablation Plan

| Run | Augmentation | Expected Test Acc |
|-----|-------------|-------------------|
| A | Current (minimal) | 86.1% (ConvNeXt) |
| B | + RandomResizedCrop + stronger ColorJitter | +1-2% |
| C | B + ElasticTransform + GridDistortion | +1-2% |
| D | C + GaussNoise + CoarseDropout | +0.5-1% |
| E | D + Hair augmentation | +0.5-1% |
| F | Full pipeline (E = recommended) | **target: 87-88%** |

---

## Success Criteria

- [ ] Test Acc@1 improves by ≥1% vs current augmentation
- [ ] Macro F1 improves (especially for minority classes)
- [ ] Training loss curve is smoother (stronger augmentation = harder task = slower but more stable)
- [ ] Train-test gap decreases (stronger regularization)
- [ ] No NaN in augmented images (`torch.isnan(batch).any()` = False)
