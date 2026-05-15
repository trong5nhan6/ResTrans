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

    # weight: [1, D] -> [D]
    w = proj.weight.squeeze()

    logits = torch.einsum('d, n b t d -> n b t', w, K)
    attn = logits.softmax(0)

    h = torch.einsum('n b t, n b t d -> b t d', attn, V)
    return h
# def block_attn_res(blocks, partial_block, proj, norm):
#     V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
#     K = norm(V)

#     # [N+1, B, T, D]
#     logits = proj(K)

#     # reduce theo feature
#     logits = logits.mean(-1)   # hoặc sum(-1)

#     # normalize logits (QUAN TRỌNG)
#     # logits = logits - logits.mean(dim=0, keepdim=True)
#     # logits = logits / (logits.std(dim=0, keepdim=True) + 1e-6)

#     attn = logits.softmax(0)

#     h = torch.einsum('n b t, n b t d -> b t d', attn, V)
#     return h

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
        self.mlp_res_proj = nn.Linear(dim, 1, bias=False)
        
        # self.attn_res_proj = nn.Linear(dim, dim, bias=True)
        # self.mlp_res_proj = nn.Linear(dim, dim, bias=True)

        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_norm = RMSNorm(dim)

        self.block_size = block_size
        self.layer_number = layer_number

        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)

    def forward(self, blocks, hidden_states):
        partial_block = hidden_states

        # ---- AttnRes before attention ----
        h = block_attn_res(blocks, partial_block,
                           self.attn_res_proj, self.attn_res_norm)

        # ---- block boundary ----
        if self.layer_number % (self.block_size // 2) == 0:
            blocks.append(partial_block)
            partial_block = None

        # ---- self-attention ----
        attn_out, _ = self.attn(self.attn_norm(h), self.attn_norm(h), self.attn_norm(h), need_weights=False)
        if partial_block is None:
            partial_block = attn_out
        else:
            partial_block = partial_block + attn_out

        # ---- AttnRes before MLP ----
        h = block_attn_res(blocks, partial_block,
                           self.mlp_res_proj, self.mlp_res_norm)

        # ---- MLP ----
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
            nn.Linear(dim, num_classes)
        )
        self.vit.heads = nn.Identity()

    def get_param_groups(self, lr_base=1e-4, lr_attn_res_norm=1e-3, lr_mlp_res_norm=1e-3, lr_head=1e-3):
        attn_res_norm_params, mlp_res_norm_params, head_params, other_params = [], [], [], []
        special_ids = set()

        for blk in self.vit.encoder.layers:
            for p in blk.attn_res_norm.parameters():
                attn_res_norm_params.append(p)
                special_ids.add(id(p))
            for p in blk.mlp_res_norm.parameters():
                mlp_res_norm_params.append(p)
                special_ids.add(id(p))

        for p in self.head.parameters():
            head_params.append(p)
            special_ids.add(id(p))

        for p in self.parameters():
            if id(p) not in special_ids:
                other_params.append(p)

        return [
            {"params": other_params,         "lr": lr_base},
            {"params": attn_res_norm_params, "lr": lr_attn_res_norm},
            {"params": mlp_res_norm_params,  "lr": lr_mlp_res_norm},
            {"params": head_params,          "lr": lr_head},
        ]

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

    # def forward(self, x):
    #     x = self.vit._process_input(x)
    #     n = x.shape[0]
    #     cls_token = self.vit.class_token.expand(n, -1, -1)
    #     x = torch.cat((cls_token, x), dim=1)
    #     x = x + self.vit.encoder.pos_embedding
    #     x = self.vit.encoder.dropout(x)

    #     # ── in header ──
    #     print(f"\n{'Layer':>7} | {'Boundary':>8} | {'len(blocks)':>11} | {'partial_block shape':>20}")
    #     print("-" * 62)
    #     print(f"{'Init':>7} | {'':>8} | {1:>11} | {str(tuple(x.shape)):>20}  ← embedding")

    #     blocks = [x]
    #     partial_block = x

    #     for blk in self.vit.encoder.layers:
    #         is_boundary = (blk.layer_number % (blk.block_size // 2) == 0)
    #         blocks, partial_block = blk(blocks, partial_block)

    #         marker = "← SAVE+RESET" if is_boundary else ""
    #         print(f"{blk.layer_number:>7} | {'YES' if is_boundary else '':>8} | "
    #               f"{len(blocks):>11} | {str(tuple(partial_block.shape)):>20}  {marker}")

    #     x = self.vit.encoder.ln(partial_block)
    #     cls = x[:, 0]
    #     out = self.head(cls)

    #     print(f"\n{'Output':>7} | {'':>8} | {'':>11} | {str(tuple(out.shape)):>20}  ← logits")
    #     return out
    
# ── RUN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    model = ViTB16_AttnRes(block_size=6, num_classes=7)
    model.eval()

    x = torch.randn(1, 3, 224, 224)   # 1 ảnh, 3 kênh, 224x224

    print("Input shape:", x.shape)

    with torch.no_grad():
        out = model(x)
    print(model)