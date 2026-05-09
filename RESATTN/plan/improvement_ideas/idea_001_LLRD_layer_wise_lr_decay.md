# Idea 001 — Layer-wise Learning Rate Decay (LLRD) for vitb16_resattn

**Priority**: P0 (Critical — implement before any other experiment)  
**Complexity**: Low  
**Compute Cost**: None (optimizer config change only)

---

## Motivation

Standard practice for fine-tuning large pretrained transformers (ViT-B/16, BERT, GPT) is to apply lower learning rates to earlier layers. This prevents **catastrophic forgetting** — the overwriting of pretrained representations that were learned on massive datasets.

The intuition: earlier transformer layers capture fundamental visual features (edges, textures) that are general and should not be changed. Later layers capture task-specific patterns. New task-specific parameters (ResAttn heads) should train faster.

---

## Current Problem

From training logs (vitb16_resattn, run 2):
- **Epoch 10**: Test Acc@1 = 79.2% (best)
- **Epoch 20**: Test Acc@1 = 77.1% (-2.1%)
- **Epoch 30**: Test Acc@1 = 73.1% (-6.1%)

This is catastrophic forgetting. The uniform LR=1e-4 is destroying the pretrained ViT-B/16 weights that were learned on ImageNet. The ViT-B/16 attention matrices were trained at scale on ImageNet — disrupting them at LR=1e-4 is like running a hammer on a precision instrument.

Specifically:
- ViT-B/16 typically fine-tuned at LR=1e-5 to 5e-5 for full fine-tuning
- ResAttn new parameters (gamma, proj, norm) need LR=1e-4 to 1e-3 to learn effectively
- Using the same LR for both is the root cause of the degradation

---

## Hypothesis

Applying layer-wise LR decay (decay=0.75) will:
1. Preserve pretrained features in early/middle ViT layers (LR ≈ 1e-6 at layer 1)
2. Allow later layers to adapt to dermoscopy features (LR ≈ 5e-5 at layer 12)
3. Allow new ResAttn parameters to learn freely (LR = 1e-4)
4. Prevent the test accuracy degradation observed after epoch 10

---

## Expected Improvement

Based on LLRD literature:
- DINOv2 fine-tuning with LLRD: +5-10% over uniform LR on specialized domains
- BEiT fine-tuning: +3-7% for fine-grained recognition
- This project specific: expect vitb16_resattn to go from 75-79% → 83-87%

The ROC-AUC (0.922) suggests the model's probability calibration is already good — LLRD should convert this latent capability into actual accuracy improvements by preventing forgetting.

---

## Risk

Low. LLRD is a well-established technique validated across hundreds of papers. The main risk is:
- Choosing wrong decay rate (mitigated by ablation)
- Too aggressive decay (early layers frozen effectively) → may underfit if dermoscopy is very different from ImageNet (unlikely)

Failure case: If `decay=0.65`, early layers may be essentially frozen (LR ≈ 1e-8) and the model can't adapt to dermoscopy color space differences. Solution: Use decay=0.75-0.85.

---

## Implementation Plan

### Step 1: Create parameter groups with LR decay

```python
def get_parameter_groups_with_llrd(model, base_lr=1e-4, llrd_decay=0.75):
    """
    Creates optimizer parameter groups with layer-wise LR decay.
    
    Args:
        model: ViTB16_AttnRes instance
        base_lr: Learning rate for new (non-pretrained) parameters
        llrd_decay: Decay factor per layer (0.75 recommended)
    
    Returns:
        List of parameter group dicts
    """
    param_groups = []
    
    # ===== New parameters (ResAttn heads, classifier) =====
    # These get the full base_lr
    new_param_names = ['attn_res_proj', 'mlp_res_proj', 'attn_res_norm', 
                       'mlp_res_norm', 'gamma_attn', 'gamma_mlp', 'head']
    new_params = []
    backbone_params = {name: param for name, param in model.named_parameters()}
    
    # ===== ViT encoder layers with decay =====
    num_layers = len(model.vit.encoder.layers)  # 12 for ViT-B/16
    
    for layer_idx in range(num_layers):
        layer_lr = base_lr * (llrd_decay ** (num_layers - layer_idx))
        layer_params = []
        
        for name, param in model.vit.encoder.layers[layer_idx].named_parameters():
            layer_params.append(param)
        
        if layer_params:
            param_groups.append({
                'params': layer_params,
                'lr': layer_lr,
                'name': f'encoder_layer_{layer_idx}'
            })
    
    # ===== Patch embedding + pos embedding =====
    embed_lr = base_lr * (llrd_decay ** (num_layers + 1))
    embed_params = (
        list(model.vit.conv_proj.parameters()) +
        [model.vit.class_token, model.vit.encoder.pos_embedding]
    )
    param_groups.append({'params': embed_params, 'lr': embed_lr, 'name': 'embedding'})
    
    # ===== New ResAttn + head parameters =====
    for name, param in model.named_parameters():
        is_backbone = any(
            name.startswith(f'vit.encoder.layers.{i}') 
            for i in range(num_layers)
        )
        is_embed = any(name.startswith(p) for p in ['vit.conv_proj', 'vit.class_token', 
                                                      'vit.encoder.pos_embedding'])
        if not is_backbone and not is_embed:
            new_params.append(param)
    
    param_groups.append({'params': new_params, 'lr': base_lr, 'name': 'new_params'})
    
    return param_groups
```

### Step 2: Update config.py

```python
# Add to config.py
LLRD_DECAY = 0.75          # Layer-wise LR decay factor
LR_BACKBONE = 1e-4         # Base LR for backbone layers (after decay)
LR_NEW_PARAMS = 1e-4       # LR for new ResAttn parameters
WARMUP_EPOCHS = 5          # Linear warmup epochs
```

### Step 3: Update train.py optimizer setup

```python
# In __main__ block, replace optimizer line:
if MODEL_NAME == 'vitb16_resattn' and USE_LLRD:
    param_groups = get_parameter_groups_with_llrd(model, base_lr=LR, llrd_decay=LLRD_DECAY)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
```

### Step 4: Add linear warmup scheduler

```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], 
                         milestones=[WARMUP_EPOCHS])
```

---

## Files Potentially Affected

- `train.py` — optimizer creation, scheduler
- `config.py` — add LLRD_DECAY, WARMUP_EPOCHS
- `model/vitb16_resattn.py` — no changes needed (structural only)

---

## Ablation Plan

Run 4 experiments (all other settings identical):

| Run | LLRD Decay | Expected Test Acc | Notes |
|-----|-----------|-------------------|-------|
| A | 1.0 (uniform) | ~75% (current) | Baseline |
| B | 0.85 | ~80% | Gentle decay |
| C | 0.75 | ~83-85% | **Recommended** |
| D | 0.65 | ~80% | May over-constrain |
| E | Freeze backbone, head only | ~78% | Upper bound of no-forgetting |

---

## Success Criteria

- [ ] Test Acc@1 at epoch 50+ is HIGHER than epoch 10
- [ ] No accuracy degradation after epoch 20
- [ ] Train loss continues to decrease without val loss explosion
- [ ] Test Acc@1 ≥ 82% by epoch 50

---

## Validation Steps

1. Log per-layer gradient norms during training (verify early layers have smaller gradients)
2. Compare test acc curves: current (degrades) vs LLRD (should plateau or improve)
3. Check that `gamma_attn` values grow smoothly (not spike then collapse)
