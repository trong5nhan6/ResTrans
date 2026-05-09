# Idea 002 — Fix Missing Residual Connections in FullAttnResBlock

**Priority**: P0 (Critical Architecture Bug)  
**Complexity**: Low  
**Compute Cost**: None (code fix only)

---

## Motivation

`resattn.py` implements `FullAttnResBlock` which wraps standard ViT blocks. However, the forward pass is missing the fundamental residual connections that make transformer blocks work. This single bug causes the model to perform at ~41-46% accuracy instead of the expected 70-80%.

---

## Current Problem

### Current code (BROKEN):

```python
class FullAttnResBlock(nn.Module):
    def forward(self, values):
        h = self.attn_res(values)              # aggregate from all prev blocks
        attn_out = self.attn(self.norm1(h))    # compute attention
        values.append(attn_out)                # ← BUG: saves only attn_out, no residual!

        h = self.mlp_res(values)
        mlp_out = self.mlp(self.norm2(h))
        values.append(mlp_out)                 # ← BUG: same problem

        return values
```

### What's wrong:

A standard transformer block computes:
```
y = x + Attention(LayerNorm(x))      ← residual connection
z = y + MLP(LayerNorm(y))            ← residual connection
```

The current code saves ONLY `attn_out` (the attention output WITHOUT the residual). This means:
1. Each block receives the raw attention output of the previous block, not the residual sum
2. The residual skip connection from input to output is LOST
3. The pretrained transformer's equilibrium (where `x + attn(LN(x)) ≈ x` at init) is destroyed
4. Gradient flow through the network is severely impaired

This is why `resattn` achieves only 41-46% despite having DINOv2 pretrained backbone.

---

## Hypothesis

Adding the correct residual connections will restore the transformer's computational integrity and allow the pretrained DINOv2 features to be utilized. Expected result: resattn should achieve 70-80% with DINOv2 backbone (matching or exceeding the broken current performance of 46%).

---

## Expected Improvement

- Current: 41.3% @Ep20, 46% @Ep40
- After fix: expect 70-80% @Ep20, 75-83% @Ep40

This is a +30-35% absolute improvement from a one-line fix.

---

## Risk

Very low. The fix adds the standard transformer residual connection that should have been there from the beginning. The ResAttn innovation (cross-block attention) is still preserved — we just add the missing `x + attn_out` term.

---

## Implementation Plan

### Fixed `FullAttnResBlock.forward()`:

```python
class FullAttnResBlock(nn.Module):
    def __init__(self, blk, dim):
        super().__init__()
        self.attn = blk.attn
        self.mlp = blk.mlp
        self.norm1 = blk.norm1
        self.norm2 = blk.norm2
        self.attn_res = FullAttnRes(dim)
        self.mlp_res = FullAttnRes(dim)
        
        # Add gating (like vitb16_resattn) for safe initialization
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_mlp = nn.Parameter(torch.zeros(1))

    def forward(self, values):
        # ===== Pre-Attn: aggregate from all previous blocks =====
        h = self.attn_res(values)
        
        # ===== Gated blend: preserve pretrained at init =====
        # At init: gamma=0, so x = last_output (pure pretrained behavior)
        last = values[-1]
        h = last + self.gamma_attn * (h - last)
        
        # ===== Standard attention with RESIDUAL =====
        attn_out = self.attn(self.norm1(h))
        out_attn = h + attn_out                  # ← FIXED: add residual
        values.append(out_attn)

        # ===== Pre-MLP: aggregate again =====
        h2 = self.mlp_res(values)
        h2 = out_attn + self.gamma_mlp * (h2 - out_attn)  # gated blend
        
        # ===== Standard MLP with RESIDUAL =====
        mlp_out = self.mlp(self.norm2(h2))
        out_mlp = h2 + mlp_out                   # ← FIXED: add residual
        values.append(out_mlp)

        return values
```

### Also fix `FullAttnResViT.forward()`:

```python
def forward(self, x):
    B, C, H, W = x.shape

    if self.is_dino:
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.interpolate_pos_encoding(x, W, H)
    else:
        x = self.vit._process_input(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed

    values = [x]  # block 0 = embedding
    for blk in self.blocks:
        values = blk(values)

    # Final aggregation
    x = self.final_res(values)
    x = self.norm(x)
    
    return x[:, 0]  # CLS token
```

### Also fix `FullAttnRes.__init__()` — replace zero init:

```python
class FullAttnRes(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # ← FIXED: small random init instead of zeros
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        # Temperature parameter for stable softmax
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(dim))
    
    def forward(self, prev_values):
        V = torch.stack(prev_values)   # [L, B, T, D]
        K = self.norm(V)
        
        scores = torch.einsum('d,lbtd->lbt', self.query, K) / self.temperature
        attn = torch.softmax(scores, dim=0)
        h = torch.einsum('lbt,lbtd->btd', attn, V)
        return h
```

---

## Files Potentially Affected

- `model/resattn.py` — primary fix (FullAttnResBlock.forward, FullAttnRes.__init__)
- No other files affected

---

## Ablation Plan

| Run | Change | Expected Result |
|-----|--------|-----------------|
| A | Current broken code | 41-46% (baseline) |
| B | Add residual, NO gating | 70-75% |
| C | Add residual + gamma gating | 75-80% |
| D | Add residual + gating + random query init | 78-83% |
| E | Add residual + gating + temperature scaling | 78-85% |

---

## Success Criteria

- [ ] Test Acc@1 at epoch 20 ≥ 70% (was 41%)
- [ ] Train loss decreases smoothly (no spikes)
- [ ] Gradient norms are stable across all layers
- [ ] Val acc tracks test acc direction (correlation coefficient > 0.6)

---

## Validation Steps

1. Run forward pass test: `torch.isnan(output).any()` → False
2. Compute gradient norms per layer, verify no vanishing/exploding
3. Train for 10 epochs, verify loss ≈ 1.4-1.8 at epoch 1 (random-init expectation)
4. Compare attention weight distributions: `attn.mean(dim=0)` should not be all-uniform
