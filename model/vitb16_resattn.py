import math
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

# =========================
# RMSNorm
# =========================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.scale


# =========================
# Block AttnRes
# =========================
def block_attn_res(blocks, partial_block, proj, norm):
    """
    blocks: list of [B, T, D]
    partial_block: [B, T, D]
    """
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)

    w = proj.weight.squeeze()  # [D]

    logits = torch.einsum('d, n b t d -> n b t', w, K) / math.sqrt(K.shape[-1])
    attn = logits.softmax(0)

    h = torch.einsum('n b t, n b t d -> b t d', attn, V)
    return h

# =========================
# AttnRes Block
# =========================
class AttnResBlock(nn.Module):
    def __init__(self, vit_block, dim, block_size, layer_number):
        super().__init__()

        # reuse pretrained modules
        self.attn = vit_block.self_attention
        self.mlp = vit_block.mlp

        self.attn_norm = vit_block.ln_1
        self.mlp_norm = vit_block.ln_2

        # new params
        self.attn_res_proj = nn.Linear(dim, 1, bias=False)
        self.mlp_res_proj  = nn.Linear(dim, 1, bias=False)

        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_norm  = RMSNorm(dim)

        # gamma gate: starts at 0 → pretrained behavior preserved at init
        self.gamma_attn = nn.Parameter(torch.zeros(1))
        self.gamma_mlp  = nn.Parameter(torch.zeros(1))

        self.block_size = block_size
        self.layer_number = layer_number

        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)

    def forward(self, blocks, hidden_states):
        partial_block = hidden_states

        # ---- Gated ResAttn before attention ----
        if len(blocks) > 0:
            h_blend = block_attn_res(blocks, partial_block,
                                     self.attn_res_proj, self.attn_res_norm)
            h = partial_block + self.gamma_attn * (h_blend - partial_block)
        else:
            h = partial_block

        # ---- self-attention (standard residual, never reset) ----
        attn_out, _ = self.attn(self.attn_norm(h), self.attn_norm(h), self.attn_norm(h))
        partial_block = partial_block + attn_out

        # ---- block boundary: save every block_size layers, do NOT reset ----
        if self.layer_number % self.block_size == 0:
            blocks.append(partial_block)

        # ---- Gated ResAttn before MLP ----
        if len(blocks) > 0:
            h_blend = block_attn_res(blocks, partial_block,
                                     self.mlp_res_proj, self.mlp_res_norm)
            h = partial_block + self.gamma_mlp * (h_blend - partial_block)
        else:
            h = partial_block

        # ---- MLP (standard residual) ----
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block


# =========================
# Full ViT + AttnRes
# =========================
class ViTB16_AttnRes(nn.Module):
    def __init__(self, block_size=4, num_classes=10):
        super().__init__()

        # load pretrained ViT-B/16
        self.vit = vit_b_16(weights="IMAGENET1K_V1")

        dim = 768
        self.block_size = block_size

        # replace encoder layers
        new_layers = nn.ModuleList()
        for i, blk in enumerate(self.vit.encoder.layers):
            new_layers.append(
                AttnResBlock(
                    blk,
                    dim=dim,
                    block_size=block_size,
                    layer_number=i + 1
                )
            )

        self.vit.encoder.layers = new_layers
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # ---- patch + pos embedding ----
        x = self.vit._process_input(x)   # [B, N, D]
        n = x.shape[0]

        cls_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        # ---- AttnRes encoder ----
        blocks = [x]   # include embedding as block 0
        partial_block = x

        for blk in self.vit.encoder.layers:
            blocks, partial_block = blk(blocks, partial_block)

        x = partial_block

        # ---- head ----
        x = self.vit.encoder.ln(x)
        cls = x[:, 0]
        out = self.head(cls)

        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViTB16_AttnRes(block_size=4, num_classes=7).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total/1e6:.2f}M")
    print(f"Trainable params: {trainable/1e6:.2f}M")

    model.eval()
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"No NaN: {not torch.isnan(out).any()}")

    model.train()
    x = torch.randn(2, 3, 224, 224).to(device)
    y = torch.randint(0, 7, (2,)).to(device)
    out = model(x)
    loss = torch.nn.CrossEntropyLoss()(out, y)
    loss.backward()
    print(f"Loss: {loss.item():.4f} | Backward OK")