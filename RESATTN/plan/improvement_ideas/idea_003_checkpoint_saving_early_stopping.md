# Idea 003 — Checkpoint Saving + Early Stopping

**Priority**: P0 (Infrastructure — must have before any experiment)  
**Complexity**: Low  
**Compute Cost**: None

---

## Motivation

Currently the training script trains for 100 epochs without saving the best model. This means:
1. The best model checkpoint is permanently lost (vitb16_resattn best was at epoch 10)
2. There is no way to recover if training overshoots
3. All 100 epochs of compute are wasted if the model overfits

---

## Current Problem

From vitb16_resattn training logs:
- Epoch 10: Test Acc@1 = **79.2%** (best)
- Epoch 100: Test Acc@1 = ~73% (much worse)

Without checkpoint saving, the best 79.2% model is gone forever. The final model (73%) is what gets evaluated.

This means the project is systematically under-reporting performance by 5-10%.

---

## Hypothesis

With proper checkpoint saving (best val Macro F1) and early stopping (patience=15 epochs on val loss), we will:
1. Always save the best model
2. Stop training when the model starts degrading
3. Reduce total training time by 30-50%
4. Improve reported accuracy by 5-8% (by using the correct checkpoint)

---

## Expected Improvement

vitb16_resattn "true best": ~79% instead of ~73-75%  
Combined with LLRD fix: could reach 83-87%

---

## Risk

None. This is pure infrastructure improvement.

---

## Implementation Plan

### Step 1: Best Model Checkpoint Saver

```python
class BestModelCheckpoint:
    """
    Saves the model checkpoint when a monitored metric improves.
    Uses Macro F1 on val set as the monitor (more reliable than accuracy for imbalanced data).
    """
    def __init__(self, save_dir, model_name, monitor='val_f1', mode='max', verbose=True):
        self.save_dir = save_dir
        self.model_name = model_name
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        os.makedirs(save_dir, exist_ok=True)
    
    def is_improvement(self, value):
        if self.mode == 'max':
            return value > self.best_value
        return value < self.best_value
    
    def save(self, model, optimizer, scheduler, epoch, metrics, config):
        """Save checkpoint with full state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'model_name': config.MODEL_NAME,
                'lr': config.LR,
                'batch_size': config.BATCH_SIZE,
            }
        }
        path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
        torch.save(checkpoint, path)
        return path
    
    def __call__(self, model, optimizer, scheduler, epoch, metrics):
        value = metrics.get(self.monitor, 0)
        
        if self.is_improvement(value):
            old_best = self.best_value
            self.best_value = value
            self.best_epoch = epoch
            path = self.save(model, optimizer, scheduler, epoch, metrics, config=None)
            if self.verbose:
                logger.info(
                    f"✓ Checkpoint saved at epoch {epoch}: "
                    f"{self.monitor} {old_best:.4f} → {value:.4f} [{path}]"
                )
            return True
        return False
```

### Step 2: Early Stopping

```python
class EarlyStopping:
    """
    Stops training when val_loss doesn't improve for `patience` epochs.
    
    Why val_loss and not val_acc?
    - Val acc is too noisy (193 samples, ±5% SE)
    - Val loss is smoother and better correlates with test performance
    """
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_state = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"⚠ Early stopping triggered after {self.patience} epochs "
                    f"without improvement. Best val_loss: {self.best_loss:.4f}"
                )
                if self.restore_best_weights and self.best_state:
                    model.load_state_dict(self.best_state)
                    logger.info("✓ Restored best weights")
        
        return self.should_stop
```

### Step 3: Integrate into Training Loop

```python
# In __main__ block:
checkpoint_dir = os.path.join(MODEL_DIR, MODEL_NAME)
checkpoint_saver = BestModelCheckpoint(
    save_dir=checkpoint_dir,
    model_name=log_name.replace('.log', ''),
    monitor='val_f1',
    mode='max'
)
early_stopper = EarlyStopping(patience=15, min_delta=0.001)

for epoch in range(EPOCHS):
    train_loss, train_acc, val_loss, val_acc = train_one_epoch(...)
    
    # ===== Evaluate on val for checkpoint selection =====
    if (epoch + 1) % 5 == 0:  # Check every 5 epochs for efficiency
        val_metrics = evaluate(model, val_loader, num_classes)
        
        # Save best checkpoint
        checkpoint_saver(model, optimizer, scheduler, epoch + 1, {
            'val_f1': val_metrics['F1'],
            'val_acc': val_acc,
            'val_loss': val_loss,
        })
        
        # Early stopping check
        if early_stopper(val_loss, model):
            logger.info(f"Training stopped early at epoch {epoch + 1}")
            break
    
    scheduler.step()

# ===== Final evaluation with best model =====
logger.info(f"Best checkpoint was at epoch {checkpoint_saver.best_epoch}")
logger.info(f"Best {checkpoint_saver.monitor}: {checkpoint_saver.best_value:.4f}")
```

### Step 4: Add Model Loading for Inference

```python
def load_best_model(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    logger.info(f"Metrics at save: {checkpoint['metrics']}")
    
    return model, checkpoint['epoch'], checkpoint['metrics']
```

---

## Files Potentially Affected

- `train.py` — main training loop integration
- `config.py` — add `MODEL_DIR`, checkpoint config
- New utility functions (can go in `utils/utils.py`)

---

## Ablation Plan

Not needed — this is a pure infrastructure fix. But verify:

| Scenario | Expected Behavior |
|----------|-------------------|
| val_f1 improves every epoch | Saves checkpoint each time |
| Training overfits | Early stopping triggers at patience=15 |
| Run interrupted | Best checkpoint preserved on disk |

---

## Success Criteria

- [ ] Best checkpoint file exists at `models/{model_name}_best.pth` after training
- [ ] Loading checkpoint restores exact same metrics
- [ ] Early stopping triggers when val_loss plateaus
- [ ] Best epoch is logged at end of training
- [ ] `test.py` uses loaded checkpoint for final evaluation
