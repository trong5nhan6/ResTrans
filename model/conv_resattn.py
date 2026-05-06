from torchvision.models import convnext_tiny


def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])  # [N, B, C, H, W]

    K = norm(V)

    # scalar attention
    w = proj.weight.squeeze()  # [C]

    logits = torch.einsum('c, n b c h w -> n b h w', w, K)
    attn = logits.softmax(0)

    h = torch.einsum('n b h w, n b c h w -> b c h w', attn, V)
    return h

class ConvNeXtAttnResBlock(nn.Module):
    def __init__(self, block, dim):
        super().__init__()
        self.block = block

        self.proj = nn.Linear(dim, 1, bias=False)
        self.norm = RMSNorm(dim)

        nn.init.zeros_(self.proj.weight)

    def forward(self, blocks, x):
        h = block_attn_res(blocks, x, self.proj, self.norm)

        out = self.block(h)

        x = x + out   # intra-block residual

        return blocks, x

class ConvNeXtStage(nn.Module):
    def __init__(self, blocks, dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtAttnResBlock(b, dim) for b in blocks
        ])

    def forward(self, x):
        blocks = []
        partial = x

        for blk in self.blocks:
            blocks, partial = blk(blocks, partial)
            blocks.append(partial)

        return partial

class CrossStageAttnRes(nn.Module):
    def __init__(self, dims, out_dim):
        super().__init__()

        self.align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, out_dim, 1),
            ) for d in dims
        ])

        self.proj = nn.Linear(out_dim, 1, bias=False)
        self.norm = RMSNorm(out_dim)

        nn.init.zeros_(self.proj.weight)

    def forward(self, features):
        target_size = features[-1].shape[-2:]

        aligned = []
        for f, align in zip(features, self.align):
            f = nn.functional.interpolate(f, size=target_size, mode='bilinear')
            f = align(f)
            aligned.append(f)

        V = torch.stack(aligned)  # [N, B, C, H, W]

        K = self.norm(V)

        w = self.proj.weight.squeeze()

        logits = torch.einsum('c, n b c h w -> n b h w', w, K)
        attn = logits.softmax(0)

        out = torch.einsum('n b h w, n b c h w -> b c h w', attn, V)

        return out


class ConvNeXt_AttnRes(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        base = convnext_tiny(weights="IMAGENET1K_V1")

        self.stem = base.features[0]

        self.stage1 = ConvNeXtStage(base.features[1], 96)
        self.stage2 = ConvNeXtStage(base.features[2], 192)
        self.stage3 = ConvNeXtStage(base.features[3], 384)
        self.stage4 = ConvNeXtStage(base.features[4], 768)

        self.cross_attn = CrossStageAttnRes(
            dims=[96, 192, 384, 768],
            out_dim=768
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.stem(x)

        e1 = self.stage1(x)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)

        x = self.cross_attn([e1, e2, e3, e4])

        x = self.pool(x).flatten(1)
        x = self.head(x)

        return x