# Task 002 — Train vitb16_resattn with Layer-wise LR Decay

**Status**: READY (after Task 001 is done)  
**Estimated Time**: 2-3 hours (coding) + 3-4 hours (100 epoch training)  
**Priority**: P1 — Highest impact single change

---

## Objective

Retrain `vitb16_resattn` with layer-wise learning rate decay (LLRD) to prevent catastrophic forgetting and recover the 79%+ performance that was achieved at epoch 10 but then lost.

---

## Exact Files to Modify

### 1. `config.py` — Add new config options

```python
# Layer-wise LR decay config
USE_LLRD = True           # Enable layer-wise LR decay
LLRD_DECAY = 0.75         # Decay factor per layer
WARMUP_EPOCHS = 5         # Linear warmup epochs
GRADIENT_CLIP = 1.0       # Max gradient norm (0 = disabled)
```

### 2. `train.py` — Add LLRD optimizer setup

Add this function before `if __name__ == "__main__":`:

```python
def build_optimizer_with_llrd(model, model_name, base_lr, weight_decay, 
                               llrd_decay=0.75):
    """
    Build AdamW optimizer with layer-wise LR decay for ViT-based models.
    
    For non-ViT models, falls back to standard optimizer.
    """
    if model_name not in ['vitb16_resattn']:
        return torch.optim.AdamW(model.parameters(), lr=base_lr, 
                                  weight_decay=weight_decay)
    
    param_groups = []
    
    # ===== New ResAttn parameters (get full base_lr) =====
    new_param_names_patterns = [
        'attn_res_proj', 'mlp_res_proj', 'attn_res_norm', 
        'mlp_res_norm', 'gamma_attn', 'gamma_mlp', 'head'
    ]
    
    # Identify all ViT encoder layer parameter names
    vit_layer_params = {}  # layer_idx -> list of params
    new_params = []
    embed_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if it's a ViT encoder layer
        matched_layer = None
        for i in range(12):  # ViT-B/16 has 12 layers
            if f'vit.encoder.layers.{i}.' in name:
                matched_layer = i
                break
        
        if matched_layer is not None:
            if matched_layer not in vit_layer_params:
                vit_layer_params[matched_layer] = []
            vit_layer_params[matched_layer].append(param)
        elif any(p in name for p in ['vit.conv_proj', 'vit.class_token', 
                                       'vit.encoder.pos_embedding',
                                       'vit.encoder.ln']):
            embed_params.append(param)
        else:
            # New ResAttn params or head
            new_params.append(param)
    
    # ===== Build param groups with LLRD =====
    num_layers = 12  # ViT-B/16
    
    for layer_idx, params in vit_layer_params.items():
        layer_lr = base_lr * (llrd_decay ** (num_layers - layer_idx))
        param_groups.append({
            'params': params,
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'name': f'encoder_layer_{layer_idx:02d}'
        })
    
    # Embedding gets very low LR
    embed_lr = base_lr * (llrd_decay ** (num_layers + 1))
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'lr': embed_lr,
            'weight_decay': weight_decay,
            'name': 'embedding'
        })
    
    # New parameters get full LR
    if new_params:
        param_groups.append({
            'params': new_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'new_params'
        })
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Log LR distribution
    print("=== LLRD Parameter Groups ===")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  {pg['name']}: lr={pg['lr']:.2e}, params={n_params/1e6:.2f}M")
    
    return optimizer
```

### 3. `train.py` — Update training loop for warmup scheduler

```python
# Replace the optimizer + scheduler setup in __main__:

if USE_LLRD and MODEL_NAME == 'vitb16_resattn':
    optimizer = build_optimizer_with_llrd(
        model, MODEL_NAME, LR, WEIGHT_DECAY, LLRD_DECAY
    )
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, 
                                   weight_decay=WEIGHT_DECAY)

# Warmup + cosine scheduler
if WARMUP_EPOCHS > 0:
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, 
                                T_max=NUM_EPOCHS - WARMUP_EPOCHS, 
                                eta_min=1e-7)
    scheduler = SequentialLR(optimizer, [warmup, cosine], 
                              milestones=[WARMUP_EPOCHS])
else:
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
```

### 4. `train.py` — Add gradient clipping to `train_one_epoch()`

```python
loss.backward()
if GRADIENT_CLIP > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
optimizer.step()
```

---

## Implementation Steps

1. Complete Task 001 first (bug fixes)
2. Update `config.py` with new variables
3. Add `build_optimizer_with_llrd()` function to `train.py`
4. Update optimizer/scheduler creation in `__main__`
5. Add gradient clipping to `train_one_epoch()`
6. Set `MODEL_NAME = "vitb16_resattn"` in config
7. Set `USE_LLRD = True`, `LLRD_DECAY = 0.75`, `WARMUP_EPOCHS = 5`
8. Set `FOCAL_LOSS = False`, use `CrossEntropy(weight=class_weights_tensor)`
9. Run training

---

## Config for This Experiment

```python
MODEL_NAME = "vitb16_resattn"
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 32

USE_LLRD = True
LLRD_DECAY = 0.75
WARMUP_EPOCHS = 5
GRADIENT_CLIP = 1.0

USE_CUTMIX = True
USE_MIXUP = True
ALPHA_MIXUP = 0.4
ALPHA_CUTMIX = 1.0
MINORITY_CLASS = None

FOCAL_LOSS = False   # USE CrossEntropy this time
```

---

## Expected Metrics

| Metric | Current (no LLRD) | Target (with LLRD) |
|--------|------------------|-------------------|
| Test Acc@1 @Ep20 | 77.1% | 82-85% |
| Test Acc@1 @Ep50 | ~73% | 83-86% |
| Test Acc@1 @Ep100 | ~73% | 84-87% |
| Macro F1 | 0.664 | 0.72-0.76 |
| Train→Test Acc gap | 0% (overfitting) | <5% (healthy) |

---

## Validation Steps

During training, monitor:
1. `val_acc` should not oscillate more than ±3% between consecutive epochs
2. `test_acc` at epoch 20 should be HIGHER than epoch 10 (not degrading)
3. Per-layer gradient norms: early layers should have smaller gradients than later layers
4. `gamma_attn` values should grow steadily from 0 to ~0.1-0.5 over training

```python
# Add to training loop for debugging:
if epoch == 0 or (epoch + 1) % 10 == 0:
    gammas = []
    for name, param in model.named_parameters():
        if 'gamma' in name:
            gammas.append((name, param.item()))
    logger.info(f"Gamma values: {gammas[:5]}")
```

---

## Rollback Plan

If LLRD causes instability:
1. Reduce `base_lr` from 1e-4 to 5e-5
2. Increase `LLRD_DECAY` from 0.75 to 0.85 (more uniform)
3. Increase warmup from 5 to 10 epochs
4. Fall back to uniform LR with just gradient clipping

---

## Log Name

```
vitb16_resattn_LLRD075_warmup5_CE_mixup_True_cutmix_True.log
```
