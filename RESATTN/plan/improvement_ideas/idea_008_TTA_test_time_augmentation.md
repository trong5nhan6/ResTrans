# Idea 008 — Test-Time Augmentation (TTA) for Free Accuracy Boost

**Priority**: P1 (Zero-cost improvement — implement for all inference)  
**Complexity**: Low  
**Compute Cost**: ×4-8 inference time (no training change)

---

## Motivation

Dermoscopy images have no canonical orientation — a melanoma looks the same at 0°, 90°, 180°, 270°. By averaging predictions across multiple augmentations, we reduce variance and improve accuracy at zero training cost.

---

## Current Problem

Single-pass inference is used: `model(x)` → `argmax` → class prediction.

This throws away the model's inherent uncertainty. The confidence on any single input orientation may be lower than the average across multiple orientations.

---

## Expected Improvement

- +1-2% Test Acc@1 (conservative estimate)
- +0.5-1% Macro F1
- Essentially FREE — only adds inference compute

---

## Implementation Plan

```python
@torch.no_grad()
def evaluate_with_tta(model, loader, num_classes, tta_transforms=None):
    """
    Evaluate model with Test-Time Augmentation.
    
    TTA strategy for dermoscopy:
    - Original
    - Horizontal flip
    - Vertical flip
    - 90° rotation
    - 180° rotation
    - 270° rotation
    - Horizontal + Vertical flip
    - 90° + Horizontal flip
    
    Total: 8 augmentations → average softmax probabilities
    """
    model.eval()
    
    if tta_transforms is None:
        import torchvision.transforms.functional as TF
        
        # Define 8 TTA transforms as lambda functions
        tta_transforms = [
            lambda x: x,                                          # original
            lambda x: TF.hflip(x),                               # horizontal flip
            lambda x: TF.vflip(x),                               # vertical flip
            lambda x: TF.rotate(x, 90),                          # 90°
            lambda x: TF.rotate(x, 180),                         # 180°
            lambda x: TF.rotate(x, 270),                         # 270°
            lambda x: TF.hflip(TF.vflip(x)),                     # H+V flip
            lambda x: TF.hflip(TF.rotate(x, 90)),                # 90° + H flip
        ]
    
    all_probs = []
    all_labels = []
    
    for imgs, labels in tqdm(loader, desc="TTA Evaluating", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        batch_probs = []
        
        for tta_fn in tta_transforms:
            # Apply TTA transform
            imgs_aug = torch.stack([tta_fn(img) for img in imgs])
            
            # Forward pass
            if MODEL_NAME == 'vit_moe':
                outputs, _ = model(imgs_aug)
            else:
                outputs = model(imgs_aug)
            
            probs = torch.softmax(outputs, dim=1)
            batch_probs.append(probs)
        
        # Average across TTA variants
        avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)  # [B, C]
        all_probs.append(avg_probs)
        all_labels.append(labels)
    
    all_probs = torch.cat(all_probs, dim=0)    # [N, C]
    all_labels = torch.cat(all_labels, dim=0)  # [N]
    
    # Convert to logits (or use probs directly for metrics)
    all_logits = all_probs.log()  # log-softmax as proxy logits
    
    metrics = compute_metrics(all_logits, all_labels, num_classes)
    return metrics
```

### Integrate into evaluation

```python
# In train.py, add TTA option
USE_TTA = True  # Add to config.py

# Final evaluation with TTA
if USE_TTA:
    test_metrics = evaluate_with_tta(model, test_loader, num_classes)
else:
    test_metrics = evaluate(model, test_loader, num_classes)
```

---

## Files Potentially Affected

- `utils/utils.py` — add `evaluate_with_tta()`
- `train.py` — call `evaluate_with_tta` at final evaluation
- `config.py` — add `USE_TTA = True`

---

## Success Criteria

- [ ] TTA test acc ≥ non-TTA test acc by ≥0.5%
- [ ] TTA inference time < 10× single-pass (8 TTA = 8×)
- [ ] No NaN in averaged probabilities
