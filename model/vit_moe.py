import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# =========================
# Dense MoE
# =========================
class DenseMoE(nn.Module):
    def __init__(self, dim, hidden_dim=512, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, hidden_dim)
            ) for _ in range(num_experts)
        ])

        self.router = nn.Linear(dim, num_experts)

    def forward(self, x):
        gate_logits = self.router(x)

        # 🔥 temperature để tránh collapse
        gate_scores = torch.softmax(gate_logits / 0.7, dim=-1)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        out = (gate_scores.unsqueeze(-1) * expert_outputs).sum(dim=1)

        return out


# =========================
# ViT Block MoE
# =========================
class ViT_BlockMoE(nn.Module):
    def __init__(self, num_classes=7, num_experts=3, block_size=4):
        super().__init__()

        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        self.patch_embed = vit.conv_proj
        self.cls_token = vit.class_token
        self.pos_embed = vit.encoder.pos_embedding
        self.encoder_layers = vit.encoder.layers

        self.hidden_dim = vit.hidden_dim
        self.block_size = block_size

        self.num_layers = len(self.encoder_layers)
        self.num_blocks = self.num_layers // block_size

        # ===== LayerNorm từng layer =====
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(self.num_layers)
        ])

        # ===== Block MoE =====
        self.block_moes = nn.ModuleList([
            DenseMoE(self.hidden_dim * block_size, hidden_dim=512, num_experts=num_experts)
            for _ in range(self.num_blocks)
        ])

        # ===== Block classifier =====
        self.block_classifiers = nn.ModuleList([
            nn.Linear(512, num_classes)   # output của MoE = 512
            for _ in range(self.num_blocks)
        ])

        # ===== Global MoE =====
        self.global_input_dim = 512 * self.num_blocks + self.hidden_dim

        self.global_moe = DenseMoE(
            self.global_input_dim,
            hidden_dim=512,
            num_experts=num_experts
        )

        # ===== Final classifier =====
        self.final_classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        # ===== Patch embedding =====
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        layer_buffer = []
        block_features = []
        block_logits = []

        # ===== Encoder =====
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)

            # CLS + token pooling
            cls = x[:, 0]
            tokens = x[:, 1:].mean(dim=1)
            feat = cls + tokens

            feat = self.layer_norms[i](feat)
            layer_buffer.append(feat)

            # ===== đủ block =====
            if (i + 1) % self.block_size == 0:
                b_idx = len(block_features)

                block_feat = torch.cat(layer_buffer, dim=1)  # [B, n*C]

                block_out = self.block_moes[b_idx](block_feat)  # [B, 512]

                logits = self.block_classifiers[b_idx](block_out)

                block_features.append(block_out)
                block_logits.append(logits)

                layer_buffer = []

        # ===== Final layer =====
        cls = x[:, 0]
        tokens = x[:, 1:].mean(dim=1)
        final_feat = cls + tokens
        final_feat = F.layer_norm(final_feat, (self.hidden_dim,))

        # ===== Fusion =====
        block_features = torch.cat(block_features, dim=1)  # [B, 512 * Bk]

        fused = torch.cat([block_features, final_feat], dim=1)
        # shape = [B, 512*Bk + 768]

        # ===== Global MoE =====
        fused = self.global_moe(fused)

        final_logits = self.final_classifier(fused)

        return final_logits, block_logits

# if __name__ == "__main__":
#     # ===== init =====
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = ViT_BlockMoE(
#         num_classes=7,
#         num_experts=2,
#         block_size=4
#     ).to(device)

#     # ===== print model =====
#     print(model)

#     # ===== count parameters =====
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f"\nTotal params: {total_params:,}")
#     print(f"Trainable params: {trainable_params:,}")

#     # ===== dummy input =====
#     x = torch.randn(2, 3, 224, 224).to(device)

#     # ===== forward =====
#     model.eval()
#     with torch.no_grad():
#         final_logits, block_logits = model(x)

#     # ===== print output shapes =====
#     print("\n=== OUTPUT SHAPES ===")
#     print("Final logits:", final_logits.shape)

#     for i, logit in enumerate(block_logits):
#         print(f"`Block {i} logits:", logit.shape)