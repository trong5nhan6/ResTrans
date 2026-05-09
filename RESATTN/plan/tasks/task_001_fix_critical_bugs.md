# Task 001 — Fix All Critical Bugs (Phase 0 — Do First)

**Status**: READY TO IMPLEMENT  
**Estimated Time**: 4-5 hours  
**Priority**: P0 — BLOCKING all other experiments

---

## Objective

Fix all identified critical bugs that currently invalidate training results. These must be done before ANY experiment.

---

## Bug 1: FullAttnResBlock Missing Residual Connections

**File**: `model/resattn.py`  
**Lines**: `FullAttnResBlock.forward()`

**Current (BROKEN)**:
```python
def forward(self, values):
    h = self.attn_res(values)
    attn_out = self.attn(self.norm1(h))
    values.append(attn_out)              # ← MISSING: should be h + attn_out

    h = self.mlp_res(values)
    mlp_out = self.mlp(self.norm2(h))
    values.append(mlp_out)              # ← MISSING: should be h2 + mlp_out
    return values
```

**Fixed**:
```python
def forward(self, values):
    # Pre-attention: aggregate from all previous blocks
    h = self.attn_res(values)
    attn_out = self.attn(self.norm1(h))
    out = h + attn_out                   # ← FIXED: standard residual
    values.append(out)

    # Pre-MLP: aggregate again (now includes attention output)
    h2 = self.mlp_res(values)
    mlp_out = self.mlp(self.norm2(h2))
    out2 = h2 + mlp_out                  # ← FIXED: standard residual
    values.append(out2)
    return values
```

**Validation**: After fix, `resattn` test acc at epoch 1 should be ~15-20% (random init expected), NOT ~13% (below random = broken).

---

## Bug 2: BlockAttnResLayer Resets partial to None

**File**: `model/block_resattn.py`  
**Lines**: `BlockAttnResLayer.forward()` block boundary logic

**Current (BROKEN)**:
```python
if (self.layer_id + 1) % self.block_size == 0:
    blocks.append(partial)
    partial = None           # ← WRONG: cuts gradient path
```

**Fixed**:
```python
if (self.layer_id + 1) % self.block_size == 0:
    blocks.append(partial)
    # DO NOT reset partial — continue accumulating residuals
```

Also fix the `None` handling in `forward()`:
```python
def forward(self, blocks, partial):
    # Remove the None-handling special case since partial is never None now
    h = self.attn_res(blocks, partial)   # partial is always a tensor
    
    attn_out = self.attn(self.norm1(h))
    partial = partial + attn_out         # always additive residual
    
    h2 = self.mlp_res(blocks, partial)
    mlp_out = self.mlp(self.norm2(h2))
    partial = partial + mlp_out
    
    if (self.layer_id + 1) % self.block_size == 0:
        blocks.append(partial)
        # Do NOT reset partial
    
    return blocks, partial
```

**Validation**: block_resattn epoch 1 loss should be ~1.8-2.2 (reasonable for 7 classes).

---

## Bug 3: CutMix In-Place Modification

**File**: `utils/utils.py`  
**Function**: `cutmix_data()`

**Current (BROKEN)**:
```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]  # ← IN-PLACE BUG
    # ...
```

**Fixed**:
```python
def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    x = x.clone()                         # ← FIXED: clone before modifying
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
```

Note: Also fixes a subtle **coordinate bug** — `rand_bbox` returns `(bbx1, bby1, bbx2, bby2)` where `bbx` is width (x-axis) and `bby` is height (y-axis). The indexing should be `x[:, :, bby1:bby2, bbx1:bbx2]` (height first, then width). The original code uses `x[:, :, bbx1:bbx2, bby1:bby2]` which is x-first, wrong for standard image tensor ordering [B, C, H, W].

Also apply same fix to `cutmix_data_class_aware()`.

---

## Bug 4: Add Gradient Clipping

**File**: `train.py`  
**Where**: Inside `train_one_epoch()` after `loss.backward()`

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # ← ADD THIS
optimizer.step()
```

---

## Bug 5: Add Seed Fixing

**File**: `train.py`  
**Where**: At the top of `if __name__ == "__main__":`

```python
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    set_seed(42)
    # ... rest of training ...
```

Add to `config.py`:
```python
SEED = 42
```

---

## Bug 6: Add Model Checkpoint Saving

**File**: `train.py`  
**Where**: Training loop, after each epoch

```python
best_val_acc = 0.0
os.makedirs(MODEL_DIR, exist_ok=True)

for epoch in range(EPOCHS):
    train_loss, train_acc, val_loss, val_acc = train_one_epoch(...)
    
    # ===== Save best model =====
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, os.path.join(MODEL_DIR, f'{MODEL_NAME}_best.pth'))
        logger.info(f"✓ Best model saved at epoch {epoch+1} (val_acc={val_acc:.4f})")
    
    scheduler.step()

logger.info(f"Training complete. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
```

---

## Validation Steps

After implementing all fixes, run these validation tests:

```python
# test_fixes.py — Add to project root
import torch
import numpy as np

def test_resattn_residuals():
    """Test that FullAttnResBlock has proper residual connections."""
    from model.resattn import FullAttnResBlock, FullAttnRes
    import torchvision.models as models
    
    # Create dummy block
    vit = models.vit_b_16(weights=None)
    blk = vit.encoder.layers[0]
    attn_res_blk = FullAttnResBlock(blk, dim=768)
    
    # Forward pass
    x = torch.randn(2, 197, 768)
    values = [x]
    values = attn_res_blk(values)
    
    # Check output is not same as input (residual should change the output)
    assert len(values) == 3  # initial + attn_out + mlp_out
    assert not torch.allclose(values[-1], x), "Residual not working!"
    print("✓ FullAttnResBlock residuals: OK")

def test_cutmix_not_inplace():
    """Test that CutMix doesn't modify the original tensor."""
    from utils.utils import cutmix_data
    
    x = torch.randn(4, 3, 224, 224)
    x_orig = x.clone()
    y = torch.randint(0, 7, (4,))
    
    x_mixed, ya, yb, lam = cutmix_data(x, y)
    
    # Original x should be unchanged
    assert torch.allclose(x, x_orig), "CutMix modified original tensor in-place!"
    print("✓ CutMix non-inplace: OK")

def test_gradient_flow():
    """Test gradient flow through FullAttnResBlock."""
    from model.resattn import FullAttnResClassifier
    
    model = FullAttnResClassifier(num_classes=7, backbone_name="vitb16")
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 7, (2,))
    
    out = model(x)
    loss = torch.nn.CrossEntropyLoss()(out, y)
    loss.backward()
    
    # Check all parameters have gradients
    no_grad = [(n, p) for n, p in model.named_parameters() 
               if p.grad is None and p.requires_grad]
    if no_grad:
        print(f"⚠ Parameters without gradient: {[n for n,p in no_grad[:5]]}")
    else:
        print("✓ Gradient flow: OK — all parameters receive gradients")

if __name__ == "__main__":
    test_resattn_residuals()
    test_cutmix_not_inplace()
    test_gradient_flow()
    print("\n✓ All critical bug fixes validated!")
```

---

## Expected Results After Fixes

| Model | Before Fix | After Fix |
|-------|-----------|-----------|
| resattn | 41-46% @Ep20 | 70-78% @Ep20 |
| block_resattn | 56-62% @Ep40 | 70-78% @Ep40 |
| vitb16_resattn | 75-79% (degrades) | 75-79% (stable) |

---

## Rollback Plan

All fixes are additive — the only fix that changes existing behavior is the partial=None removal in block_resattn. If this breaks something:
1. Re-add `partial = None` reset but keep it as a config option
2. The resattn fix only adds `h + attn_out` — purely additive, safe to rollback
3. CutMix fix only adds `.clone()` — no behavioral change if original wasn't being reused
