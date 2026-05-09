# Idea 005 — LDAM Loss for Minority Class Improvement

**Priority**: P2  
**Complexity**: Low  
**Compute Cost**: Negligible

---

## Motivation

The ISIC2018 dataset has extreme class imbalance (66:1 ratio between NV and DF). Despite using WeightedRandomSampler, the model still struggles with minority classes:
- Current F1: 0.664 (macro) — implies some classes have F1 < 0.5
- ROC-AUC: 0.922 — good ranking, but poor decision boundary for minority classes

LDAM (Label-Distribution-Aware Margin) specifically designs larger decision margins for minority classes, making the classifier more conservative about classifying samples into rare categories.

---

## Current Problem

The model's ROC-AUC = 0.922 suggests it has reasonable probability estimates, but Macro F1 = 0.664 suggests the decision boundary is biased toward NV. Specifically:
- NV (Class 1, 67% of data) gets classified correctly most of the time
- DF (Class 5, 1.15%) and VASC (Class 6, 1.42%) are likely misclassified as NV or MEL

WeightedRandomSampler helps by oversampling minorities in each batch, but the loss function itself doesn't penalize minority class errors more heavily.

---

## Hypothesis

LDAM loss with adjusted margins will push the decision boundary away from minority classes, improving their recall at a small cost to majority class accuracy.

---

## Expected Improvement

- Macro F1: +3-5% (from 0.664 → 0.71-0.72)
- Minority class Recall: +10-20% for DF and VASC
- Overall Acc: ±1% (may decrease slightly for NV, increase for minorities)

---

## Implementation Plan

```python
class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss.
    Reference: Cao et al., NeurIPS 2019.
    
    Args:
        cls_num_list: list of sample counts per class [n_0, n_1, ..., n_C-1]
        max_m: maximum margin value (0.5 recommended)
        weight: optional per-class weight tensor
        s: scale factor (30 for cosine classifier, 1 for standard)
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        
        # Compute per-class margins: m_i ∝ n_i^{-1/4}
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        
        # Register as buffer (not trained)
        self.m_list = torch.FloatTensor(m_list)
        
        self.s = s
        self.weight = weight
    
    def forward(self, x, target):
        """
        x: logits [B, C]
        target: class indices [B]
        """
        # Get the margin for each sample's true class
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), 1)
        
        m_list = self.m_list.to(x.device)
        
        # Subtract margin from the true class logit
        index_float = index.float()
        batch_m = torch.matmul(m_list.unsqueeze(0), index_float.t())  # [1, B]
        batch_m = batch_m.view((-1, 1))
        
        x_m = x - batch_m  # reduce logit for true class → harder to classify correctly
        
        # Apply scale
        output = torch.where(index, x_m, x) * self.s
        
        return F.cross_entropy(output, target, weight=self.weight)
```

### Deferred Re-Weighting Strategy

For best results, combine LDAM with deferred re-weighting:
- **Phase 1 (first 70% of epochs)**: Train with instance-balanced sampling (no WeightedRandomSampler)
- **Phase 2 (last 30% of epochs)**: Switch to class-balanced sampling + LDAM with weights

```python
# In config.py
LDAM_WARMUP_RATIO = 0.7  # first 70% epochs: instance-balanced

# In training loop
if epoch < int(NUM_EPOCHS * LDAM_WARMUP_RATIO):
    # Instance-balanced (regular) training
    criterion = nn.CrossEntropyLoss()
else:
    # Class-balanced with LDAM
    criterion = LDAMLoss(cls_num_list=class_count.tolist(), max_m=0.5, s=1)
```

---

## Files Potentially Affected

- `utils/utils.py` — add `LDAMLoss` class
- `train.py` — modify criterion selection + deferred re-weighting logic
- `config.py` — add `USE_LDAM`, `LDAM_MAX_M`, `LDAM_WARMUP_RATIO`

---

## Ablation Plan

| Run | Loss | Sampling | Expected Macro F1 |
|-----|------|---------|-------------------|
| A | FocalLoss(gamma=2) | WeightedSampler | 0.664 (current) |
| B | CrossEntropy+weight | WeightedSampler | 0.68 |
| C | LDAM(max_m=0.5) | WeightedSampler | 0.70 |
| D | LDAM(max_m=0.5) | Deferred reweighting | **0.72-0.74** |
| E | Balanced Softmax | Instance-balanced | 0.70 |

---

## Success Criteria

- [ ] Macro F1 ≥ 0.72 (from 0.664)
- [ ] DF (Class 5) and VASC (Class 6) recall ≥ 0.5 (from likely ~0.3)
- [ ] Overall Test Acc@1 does not drop by more than 1%
- [ ] ROC-AUC stays ≥ 0.90
