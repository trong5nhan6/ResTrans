# Code Quality Review

**Date**: 2026-05-09  
**Reviewer**: AI Research Lab Assistant  
**Scope**: Full codebase — `train.py`, `config.py`, `model/`, `utils/`, `data/`

---

## 1. Maintainability

### Score: 5/10

**Issues**:

#### 1.1 Global Variables in train.py
```python
# Current: optimizer, criterion, scheduler are module-level globals
# Used inside train_one_epoch() without being passed as arguments
optimizer.zero_grad()   # Where does optimizer come from?
optimizer.step()
scheduler.step()
```
`optimizer`, `criterion`, `scheduler` are defined in `__main__` but accessed inside `train_one_epoch()`. This is not a function parameter — it relies on Python's global scope lookup. This means:
- `train_one_epoch()` is not unit-testable without running `__main__`
- Moving the function to another module will break it
- Multiple concurrent training runs on the same process will share state

**Fix**: Pass optimizer and criterion as parameters to `train_one_epoch()`.

#### 1.2 Duplicate import
```python
import numpy as np  # imported twice in train.py
```

#### 1.3 Magic numbers throughout code
```python
# In vit_moe.py
gate_scores = torch.softmax(gate_logits / 0.7, dim=-1)  # 0.7 is magic number
```

#### 1.4 Commented-out dead code
Multiple blocks of commented code in train.py that should be removed or properly documented:
```python
# criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# criterion = FocalLoss(num_classes=num_classes, gamma=1.0, weight=class_weights_tensor)
```

**Score breakdown**:
- Readability: 7/10 (code is readable, comments are present)
- Structure: 4/10 (globals, no dependency injection)
- Dead code: 5/10 (moderate amount)

---

## 2. Modularity

### Score: 6/10

**What's good**:
- Each model is in its own file ✓
- `utils.py` contains utilities separate from training ✓
- `data/` contains dataset classes separate from models ✓

**Issues**:

#### 2.1 utils.py is a monolithic file
Contains: augmentation, loss functions, metrics, logging, model complexity. Should be split into:
```
utils/
├── losses.py        (FocalLoss, LDAMLoss, BalancedSoftmax)
├── metrics.py       (compute_metrics, accuracy_topk)
├── transforms.py    (get_transform, SyntheticHairAug)
├── logging_utils.py (setup_logger, log_dataset_info)
└── train_utils.py   (cutmix, mixup, checkpoint)
```

#### 2.2 No model registry
Adding a new model requires editing `train.py` directly:
```python
if MODEL_NAME == 'resattn':
    model = ...
elif MODEL_NAME == 'block_resattn':
    model = ...
# ... etc
```
Should use a registry pattern:
```python
MODEL_REGISTRY = {
    'resattn': FullAttnResClassifier,
    'block_resattn': BlockAttnResClassifier,
    # ...
}
model = MODEL_REGISTRY[MODEL_NAME](num_classes=num_classes)
```

---

## 3. Reproducibility

### Score: 4/10

**Critical Issues**:

#### 3.1 No seed fixing
Results are not reproducible between runs:
```python
# MISSING in train.py:
torch.manual_seed(SEED)
numpy.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
```

#### 3.2 DataLoader workers with different seeds
```python
# Current:
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                          sampler=sampler, num_workers=NUM_WORKERS)
# Missing:
generator = torch.Generator()
generator.manual_seed(SEED)
train_loader = DataLoader(..., generator=generator,
                          worker_init_fn=lambda wid: np.random.seed(SEED + wid))
```

#### 3.3 No experiment config saved to disk
When reviewing old logs, there's no way to know which config was used. Solution: serialize config to JSON at run start.

```python
import json
config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
with open(os.path.join(LOG_DIR, log_name.replace('.log', '_config.json')), 'w') as f:
    json.dump(config_dict, f, indent=2, default=str)
```

#### 3.4 DINOv2 downloads from internet
`torch.hub.load("facebookresearch/dinov2", ...)` depends on internet connectivity. Should cache locally:
```python
os.environ['TORCH_HOME'] = './pretrained_models'
```

---

## 4. Scalability

### Score: 5/10

**Issues**:

#### 4.1 No mixed precision (AMP)
Training is 1.5-2× slower without fp16:
```python
# MISSING: 
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(imgs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 4.2 No multi-GPU support
With ViT-B/16 (86.6M params) at batch size 32, training fits on a single GPU but could be 4× faster on multi-GPU:
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    # Or better: DistributedDataParallel
```

#### 4.3 DataLoader num_workers=4 is hardcoded
Should be adaptive based on CPU cores:
```python
NUM_WORKERS = min(4, os.cpu_count() // 2)
```

---

## 5. Experiment Tracking

### Score: 3/10

**Critical Gaps**:

#### 5.1 No TensorBoard or W&B logging
All metrics go to text files only. Loss curves, LR schedules, gradient norms are completely invisible.

```python
# Add TensorBoard:
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, 'tensorboard', MODEL_NAME))

# In training loop:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Acc/train', train_acc, epoch)
writer.add_scalar('Acc/val', val_acc, epoch)
writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
```

#### 5.2 No gradient norm logging
Can't detect gradient explosion without logging it:
```python
grad_norm = sum(p.grad.norm().item() ** 2 
                for p in model.parameters() 
                if p.grad is not None) ** 0.5
logger.info(f"Gradient norm: {grad_norm:.4f}")
```

---

## 6. Checkpoint Strategy

### Score: 2/10

**Current**: No checkpoint saving whatsoever.

**Required**:
- Save best model by val Macro F1 (not val Acc — val set is too noisy)
- Save last model (for resume)
- Save optimizer + scheduler state (for resume)
- Auto-detect and resume from checkpoint if exists

---

## 7. Logging Quality

### Score: 6/10

**What's good**:
- Timestamps in every log line ✓
- Dataset info logged at start ✓
- Model complexity logged at end ✓

**Issues**:
- Logger name "ISIC_Logger" is fixed — running two models in the same process shares logger
- Fix: `logger = logging.getLogger(f"ISIC_{MODEL_NAME}_{os.getpid()}")`
- No confusion matrix logging
- No per-class accuracy/recall logging (only macro averages)

---

## 8. Mixed Precision

### Score: 2/10

Not implemented. For ViT-B/16 at batch size 32, this is a significant bottleneck.

Expected speedup: 1.5-2× training, same accuracy.

---

## 9. Memory Efficiency

### Score: 5/10

**Issues**:

#### 9.1 `compute_metrics` stores all logits and labels in memory
```python
all_logits = []
for imgs, labels in loader:
    all_logits.append(outputs)  # accumulates all batches in RAM
all_logits = torch.cat(all_logits, dim=0)  # may cause OOM on large datasets
```
For test set (1512 samples), this is fine. For larger datasets, use running statistics.

#### 9.2 `FullAttnResViT` stores ALL previous block outputs
```python
values = [x]
for blk in self.blocks:
    values = blk(values)
# values grows to [B, 2*num_blocks+1, T, D] — all in GPU memory
```
For ViT-B/16 with 12 blocks: `values` has up to 25 tensors of shape [B, 197, 768]. At B=32: 25 × 32 × 197 × 768 × 4 bytes ≈ **4.8 GB per forward pass**.

This is a serious memory concern. Block-sparse version (block_resattn) is more memory-efficient.

---

## 10. Code Quality Checklist

| Category | Status | Score |
|----------|--------|-------|
| Readability | OK with minor issues | 7/10 |
| Global variables | Problem | 4/10 |
| Reproducibility | Missing | 4/10 |
| Checkpoint saving | Missing | 2/10 |
| Experiment tracking | Missing | 3/10 |
| Mixed precision | Missing | 2/10 |
| Multi-GPU | Missing | N/A |
| Memory efficiency | Concerns | 5/10 |
| Error handling | Minimal | 5/10 |
| Test coverage | None | 0/10 |
| Documentation | Moderate | 6/10 |
| Dead code | Present | 5/10 |

**Overall Code Quality Score: 4.5/10**

---

## 11. Quick Wins for Code Quality (< 1 hour each)

```python
# 1. Fix duplicate import (train.py line 8 and 14)
# Remove second: import numpy as np

# 2. Fix global variable issue
def train_one_epoch(model, train_loader, val_loader, optimizer, criterion,
                    use_mixup=True, ...):
    # Now optimizer and criterion are parameters, not globals

# 3. Add seed fixing at top of __main__
set_seed(SEED)

# 4. Add config serialization
save_config_to_json(config, LOG_DIR, log_name)

# 5. Fix logger name collision
logger = logging.getLogger(f"{MODEL_NAME}_{time.strftime('%H%M%S')}")
```
