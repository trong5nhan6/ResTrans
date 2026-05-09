import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models


# =========================
# RMSNorm (Conv)
# =========================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [N,B,C,H,W] hoặc [B,C,H,W]
        if x.dim() == 5:
            norm = x.pow(2).mean(dim=2, keepdim=True)
            scale = self.scale.view(1, 1, -1, 1, 1)
        else:
            norm = x.pow(2).mean(dim=1, keepdim=True)
            scale = self.scale.view(1, -1, 1, 1)

        return x * torch.rsqrt(norm + self.eps) * scale


# =========================
# Block Attention Residual
# =========================
def block_attn_res(blocks, partial, proj, norm):
    if len(blocks) == 0:
        return partial

    V = torch.stack(blocks + [partial])   # [N,B,C,H,W]
    K = norm(V)

    w = proj.weight.squeeze()  # [C]

    logits = torch.einsum('c, n b c h w -> n b h w', w, K)
    logits = logits / math.sqrt(K.shape[2])

    attn = logits.softmax(0)

    out = torch.einsum('n b h w, n b c h w -> b c h w', attn, V)
    return out


# =========================
# AttnRes Block
# =========================
class ConvNeXtAttnResBlock(nn.Module):
    def __init__(self, block, dim):
        super().__init__()
        self.block = block

        self.proj = nn.Linear(dim, 1, bias=False)
        self.norm = RMSNorm(dim)

        self.gamma = nn.Parameter(torch.zeros(1))  # gate

        nn.init.zeros_(self.proj.weight)

    def forward(self, blocks, x):
        h = block_attn_res(blocks, x, self.proj, self.norm)

        # gated residual (giữ pretrained)
        h = x + self.gamma * (h - x)

        out = self.block(h)

        x = x + out
        return x


# =========================
# Stage
# =========================
class ConvNeXtStage(nn.Module):
    def __init__(self, blocks, dim, block_size=2):
        super().__init__()

        self.blocks = nn.ModuleList([
            ConvNeXtAttnResBlock(b, dim) for b in blocks
        ])

        self.block_size = block_size

    def forward(self, x):
        blocks = []
        partial = x

        for i, blk in enumerate(self.blocks):
            partial = blk(blocks, partial)

            # block boundary
            if (i + 1) % self.block_size == 0:
                blocks.append(partial)

        # nếu block cuối chưa append
        if len(self.blocks) % self.block_size != 0:
            blocks.append(partial)

        return partial


# =========================
# Cross-stage AttnRes
# =========================
class CrossStageAttnRes(nn.Module):
    def __init__(self, dims, out_dim):
        super().__init__()

        self.align = nn.ModuleList([
            nn.Conv2d(d, out_dim, 1) for d in dims
        ])

        self.proj = nn.Linear(out_dim, 1, bias=False)
        self.norm = RMSNorm(out_dim)

        self.gamma = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.proj.weight)

    def forward(self, features):
        target_size = features[-1].shape[-2:]

        aligned = []
        for f, align in zip(features, self.align):
            f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            f = align(f)
            aligned.append(f)

        V = torch.stack(aligned)  # [N,B,C,H,W]
        K = self.norm(V)

        w = self.proj.weight.squeeze()

        logits = torch.einsum('c, n b c h w -> n b h w', w, K)
        logits = logits / math.sqrt(K.shape[2])

        attn = logits.softmax(0)

        out = torch.einsum('n b h w, n b c h w -> b c h w', attn, V)

        return features[-1] + self.gamma * (out - features[-1])


# =========================
# FULL MODEL
# =========================
class ConvNeXt_AttnRes(nn.Module):
    def __init__(self, num_classes=10, block_size=2):
        super().__init__()

        base = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )

        # ===== đúng pipeline =====
        self.stem = base.features[0]

        # Stage 1
        self.stage1 = ConvNeXtStage(base.features[1], 128, block_size)
        self.down1 = base.features[2]

        # Stage 2
        self.stage2 = ConvNeXtStage(base.features[3], 256, block_size)
        self.down2 = base.features[4]

        # Stage 3
        self.stage3 = ConvNeXtStage(base.features[5], 512, block_size)
        self.down3 = base.features[6]

        # Stage 4 (không có down sau)
        self.stage4 = ConvNeXtStage(base.features[7], 1024, block_size)

        # Cross-stage AttnRes
        self.cross_attn = CrossStageAttnRes(
            dims=[128, 256, 512, 1024],
            out_dim=1024
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)

        # Stage 1
        e1 = self.stage1(x)
        x = self.down1(e1)

        # Stage 2
        e2 = self.stage2(x)
        x = self.down2(e2)

        # Stage 3
        e3 = self.stage3(x)
        x = self.down3(e3)

        # Stage 4
        e4 = self.stage4(x)

        # Cross-stage attention
        x = self.cross_attn([e1, e2, e3, e4])

        x = self.pool(x).flatten(1)
        x = self.head(x)

        return x


# ===== import model của bạn =====
# from your_file import ConvNeXt_AttnRes

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== khởi tạo model =====
model = ConvNeXt_AttnRes(num_classes=10).to(device)

print("\n===== MODEL INFO =====")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total params: {total_params/1e6:.2f}M")
print(f"Trainable params: {trainable_params/1e6:.2f}M")


# =========================
# 1. FORWARD TEST
# =========================
print("\n===== FORWARD TEST =====")

model.eval()
x = torch.randn(2, 3, 224, 224).to(device)

with torch.no_grad():
    out = model(x)

print("Output shape:", out.shape)
print("Output sample:", out[0])


# =========================
# 2. CHECK NaN
# =========================
print("\n===== NaN CHECK =====")

if torch.isnan(out).any():
    print("❌ NaN detected!")
else:
    print("✅ No NaN, model stable")


# =========================
# 3. BACKWARD TEST
# =========================
print("\n===== BACKWARD TEST =====")

model.train()

x = torch.randn(2, 3, 224, 224).to(device)
y = torch.randint(0, 10, (2,)).to(device)

criterion = nn.CrossEntropyLoss()

out = model(x)
loss = criterion(out, y)

loss.backward()

print("Loss:", loss.item())
print("✅ Backward OK")


# =========================
# 4. GRADIENT CHECK
# =========================
print("\n===== GRADIENT CHECK =====")

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} | grad mean: {param.grad.abs().mean():.6f}")
        break


# =========================
# 5. SHAPE DEBUG (optional)
# =========================
print("\n===== SHAPE DEBUG =====")

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            print(f"{name}: {output.shape}")
    return hook

# hook vào stage
model.stage1.register_forward_hook(hook_fn("stage1"))
model.stage2.register_forward_hook(hook_fn("stage2"))
model.stage3.register_forward_hook(hook_fn("stage3"))
model.stage4.register_forward_hook(hook_fn("stage4"))

with torch.no_grad():
    _ = model(torch.randn(1, 3, 224, 224).to(device))