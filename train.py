import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import (
    DATA_ROOT, LOG_DIR, MODEL_DIR,
    BATCH_SIZE, NUM_WORKERS,
    LR, WEIGHT_DECAY,
    USE_LLRD, LLRD_DECAY,
    WARMUP_EPOCHS,
    GRADIENT_CLIP,
    NUM_EPOCHS,
    MODEL_NAME,
    USE_CUTMIX, USE_MIXUP, ALPHA_MIXUP, ALPHA_CUTMIX, MINORITY_CLASS,
    FOCAL_LOSS, GAMMA, ALPHA, REDUCTION,
    DEVICE, SEED,
    SAVE_BEST, EARLY_STOP, EARLY_PATIENCE,
)
from data.ISIC2018 import ISIC2018
from model.basemodel import BaseModel
from model.block_resattn import BlockAttnResClassifier
from model.resattn import FullAttnResClassifier
from model.vit_moe import ViT_BlockMoE
from model.vitb16_resattn import ViTB16_AttnRes
from model.conv_resattn import ConvNeXt_AttnRes
from utils.utils import (
    setup_logger, log_dataset_info,
    compute_model_complexity, compute_metrics,
    get_transform, FocalLoss,
    cutmix_data, mixup_data, mixup_criterion,
    cutmix_data_class_aware, mixup_data_class_aware,
    count_params,
)


# ──────────────────────────────────────────────
# 0. Reproducibility
# ──────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# 1. LLRD optimizer (chỉ dùng cho vitb16_resattn)
# ──────────────────────────────────────────────
def build_optimizer(model, model_name, base_lr, weight_decay,
                    use_llrd=True, llrd_decay=0.75):
    """
    Với vitb16_resattn: áp dụng Layer-wise LR Decay.
      - New ResAttn params (gamma, proj, norm): base_lr
      - ViT encoder layer i (pretrained): base_lr * llrd_decay^(num_layers-1-i)
      - Embedding / patch proj / pos_embed: base_lr * llrd_decay^num_layers  (rất nhỏ)
      - Head: base_lr

    Với các model khác: AdamW thông thường.
    """
    if model_name != "vitb16_resattn" or not use_llrd:
        return torch.optim.AdamW(
            model.parameters(), lr=base_lr, weight_decay=weight_decay
        )

    # Tên sub-string để nhận biết NEW params bên trong mỗi AttnResBlock
    NEW_KEYWORDS = (
        "attn_res_proj", "mlp_res_proj",
        "attn_res_norm", "mlp_res_norm",
        "gamma_attn",    "gamma_mlp",
    )

    num_layers = len(model.vit.encoder.layers)   # 12 với ViT-B/16

    # Gom params
    pretrained_by_layer = {i: [] for i in range(num_layers)}
    new_resattn_params  = []
    embed_params        = []
    head_params         = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Xác định layer index nếu thuộc encoder
        layer_idx = None
        for i in range(num_layers):
            if f"vit.encoder.layers.{i}." in name:
                layer_idx = i
                break

        if layer_idx is not None:
            if any(kw in name for kw in NEW_KEYWORDS):
                new_resattn_params.append(param)
            else:
                pretrained_by_layer[layer_idx].append(param)

        elif any(s in name for s in (
            "vit.conv_proj",
            "vit.class_token",
            "vit.encoder.pos_embedding",
            "vit.encoder.ln",
        )):
            embed_params.append(param)

        else:
            head_params.append(param)

    param_groups = []

    # Pretrained encoder layers — LLRD
    for i in range(num_layers):
        if pretrained_by_layer[i]:
            lr_i = base_lr * (llrd_decay ** (num_layers - 1 - i))
            param_groups.append(dict(
                params=pretrained_by_layer[i],
                lr=lr_i,
                weight_decay=weight_decay,
                name=f"pretrained_layer_{i:02d}",
            ))

    # New ResAttn params — luôn dùng base_lr
    if new_resattn_params:
        param_groups.append(dict(
            params=new_resattn_params,
            lr=base_lr,
            weight_decay=weight_decay,
            name="new_resattn",
        ))

    # Embedding — LR rất nhỏ
    if embed_params:
        embed_lr = base_lr * (llrd_decay ** num_layers)
        param_groups.append(dict(
            params=embed_params,
            lr=embed_lr,
            weight_decay=weight_decay,
            name="embed",
        ))

    # Head
    if head_params:
        param_groups.append(dict(
            params=head_params,
            lr=base_lr,
            weight_decay=weight_decay,
            name="head",
        ))

    optimizer = torch.optim.AdamW(param_groups)

    # Log để kiểm tra phân bổ LR
    print("=== LLRD Parameter Groups ===")
    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"  [{pg['name']}] lr={pg['lr']:.2e}  params={n/1e6:.3f}M")

    return optimizer


# ──────────────────────────────────────────────
# 2. Scheduler: Linear warmup → CosineAnnealing
# ──────────────────────────────────────────────
def build_scheduler(optimizer, num_epochs, warmup_epochs):
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=1e-7,
        )
        return SequentialLR(optimizer, [warmup, cosine],
                            milestones=[warmup_epochs])
    return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


# ──────────────────────────────────────────────
# 3. Train one epoch
# ──────────────────────────────────────────────
def train_one_epoch(
    model, train_loader, val_loader,
    optimizer, criterion,
    use_mixup=USE_MIXUP, use_cutmix=USE_CUTMIX,
    alpha_mixup=ALPHA_MIXUP, alpha_cutmix=ALPHA_CUTMIX,
    minority_class=MINORITY_CLASS,
    gradient_clip=GRADIENT_CLIP,
):
    """
    optimizer và criterion được truyền vào qua tham số — không còn dùng global.
    """
    model.train()
    train_loss = train_correct = train_total = 0

    for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # ── Augmentation trong batch ────────────────────────────────────────
        r = np.random.rand()
        if use_mixup and r < 0.5:
            fn = mixup_data_class_aware if minority_class else mixup_data
            imgs, targets_a, targets_b, lam = fn(
                imgs, labels, alpha=alpha_mixup,
                **({} if not minority_class else {"minority_classes": minority_class})
            )
        elif use_cutmix:
            fn = cutmix_data_class_aware if minority_class else cutmix_data
            imgs, targets_a, targets_b, lam = fn(
                imgs, labels, alpha=alpha_cutmix,
                **({} if not minority_class else {"minority_classes": minority_class})
            )
        else:
            targets_a = targets_b = labels
            lam = 1.0

        # ── Forward ─────────────────────────────────────────────────────────
        outputs = model(imgs)

        # ── Loss ─────────────────────────────────────────────────────────────
        if MODEL_NAME == "vit_moe":
            final_logits, block_logits = outputs
            if lam < 1.0:
                main_loss = mixup_criterion(criterion, final_logits,
                                            targets_a, targets_b, lam)
                aux_loss  = sum(
                    mixup_criterion(criterion, lg, targets_a, targets_b, lam)
                    for lg in block_logits
                ) / len(block_logits)
            else:
                main_loss = criterion(final_logits, labels)
                aux_loss  = sum(criterion(lg, labels)
                                for lg in block_logits) / len(block_logits)
            loss = main_loss + 0.3 * aux_loss
        else:
            final_logits = outputs
            if lam < 1.0:
                loss = mixup_criterion(criterion, outputs,
                                       targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)

        # ── Backward ─────────────────────────────────────────────────────────
        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        # ── Tracking ─────────────────────────────────────────────────────────
        train_loss += loss.item()
        preds       = final_logits.argmax(dim=1)

        if lam < 1.0:
            train_correct += (
                lam       * (preds == targets_a).sum().item() +
                (1 - lam) * (preds == targets_b).sum().item()
            )
        else:
            train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc   = train_correct / train_total

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)

            if MODEL_NAME == "vit_moe":
                final_logits, block_logits = outputs
                main_loss = criterion(final_logits, labels)
                aux_loss  = sum(criterion(lg, labels)
                                for lg in block_logits) / len(block_logits)
                loss = main_loss + 0.3 * aux_loss
            else:
                final_logits = outputs
                loss         = criterion(outputs, labels)

            val_loss    += loss.item()
            preds        = final_logits.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_loss /= len(val_loader)
    val_acc   = val_correct / val_total

    return train_loss, train_acc, val_loss, val_acc


# ──────────────────────────────────────────────
# 4. Evaluate
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    all_logits = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        if MODEL_NAME == "vit_moe":
            outputs, _ = model(imgs)
        else:
            outputs = model(imgs)

        all_logits.append(outputs)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return compute_metrics(all_logits, all_labels, num_classes)


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────
if __name__ == "__main__":

    set_seed(SEED)
    print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}  |  Seed: {SEED}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_dataset = ISIC2018(os.path.join(DATA_ROOT, "train"),
                             transform=get_transform(is_train=True))
    val_dataset   = ISIC2018(os.path.join(DATA_ROOT, "val"),
                             transform=get_transform(is_train=False))
    test_dataset  = ISIC2018(os.path.join(DATA_ROOT, "test"),
                             transform=get_transform(is_train=False))

    # ── Class imbalance ───────────────────────────────────────────────────────
    train_labels  = np.array(train_dataset.labels)
    class_count   = np.bincount(train_labels)
    num_classes   = len(class_count)
    print("Class distribution (train):", class_count)

    class_weights   = 1.0 / class_count.astype(float)
    sample_weights  = [class_weights[l] for l in train_labels]
    sampler         = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )
    class_weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32
    ).to(DEVICE)

    # ── DataLoader ────────────────────────────────────────────────────────────
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, generator=g,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if MODEL_NAME == "resattn":
        model = FullAttnResClassifier(num_classes=num_classes)
    elif MODEL_NAME == "block_resattn":
        model = BlockAttnResClassifier(num_classes=num_classes)
    elif MODEL_NAME == "vit_moe":
        model = ViT_BlockMoE(num_classes=num_classes)
    elif MODEL_NAME == "vitb16_resattn":
        model = ViTB16_AttnRes(block_size=4, num_classes=num_classes)
    elif MODEL_NAME == "conv_resattn":
        model = ConvNeXt_AttnRes(num_classes=num_classes)
    else:
        model = BaseModel(model_name=MODEL_NAME, num_classes=num_classes)

    model = model.to(DEVICE)
    print(f"Total parameters: {count_params(model)/1e6:.2f}M")

    # ── Loss ──────────────────────────────────────────────────────────────────
    if FOCAL_LOSS:
        criterion = FocalLoss(gamma=GAMMA, alpha=ALPHA, reduction=REDUCTION)
    else:
        # CrossEntropy + class weights → xử lý imbalance ổn định hơn Focal khi không có alpha
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ── Optimizer + Scheduler ────────────────────────────────────────────────
    optimizer = build_optimizer(
        model, MODEL_NAME, LR, WEIGHT_DECAY,
        use_llrd=USE_LLRD, llrd_decay=LLRD_DECAY,
    )
    scheduler = build_scheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)

    # ── Logger ────────────────────────────────────────────────────────────────
    log_name = (
        f"{MODEL_NAME}"
        f"_LLRD{LLRD_DECAY if USE_LLRD else 'off'}"
        f"_mixup{USE_MIXUP}_cutmix{USE_CUTMIX}"
        f"_focal{FOCAL_LOSS}.log"
    )
    logger = setup_logger(log_dir=LOG_DIR, log_name=log_name)

    log_dataset_info(logger, train_dataset, "Train")
    log_dataset_info(logger, val_dataset,   "Validation")
    log_dataset_info(logger, test_dataset,  "Test")

    logger.info(f"Model        : {MODEL_NAME}")
    logger.info(f"LR           : {LR}  |  LLRD={USE_LLRD}  decay={LLRD_DECAY}")
    logger.info(f"Loss         : {'FocalLoss' if FOCAL_LOSS else 'CrossEntropy+weight'}")
    logger.info(f"Warmup       : {WARMUP_EPOCHS} epochs")
    logger.info(f"Gradient clip: {GRADIENT_CLIP}")
    logger.info(f"Seed         : {SEED}")

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path    = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.pth")
    best_val_acc = 0.0
    best_epoch   = 0

    # ── Early stopping state ──────────────────────────────────────────────────
    es_counter   = 0
    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc, val_loss, val_acc = train_one_epoch(
            model, train_loader, val_loader,
            optimizer, criterion,
            use_mixup=USE_MIXUP,
            use_cutmix=USE_CUTMIX,
            alpha_mixup=ALPHA_MIXUP,
            alpha_cutmix=ALPHA_CUTMIX,
            minority_class=MINORITY_CLASS,
            gradient_clip=GRADIENT_CLIP,
        )

        scheduler.step()

        logger.info(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}]  "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")

        # ── Save best checkpoint ──────────────────────────────────────────────
        if SAVE_BEST and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_acc":         val_acc,
                "val_loss":        val_loss,
            }, ckpt_path)
            logger.info(f"  ✓ Best model saved  (val_acc={val_acc:.4f}  epoch={epoch+1})")

        # ── Early stopping ────────────────────────────────────────────────────
        if EARLY_STOP:
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                es_counter    = 0
            else:
                es_counter += 1
                if es_counter >= EARLY_PATIENCE:
                    logger.info(
                        f"  ⚠ Early stopping triggered at epoch {epoch+1} "
                        f"(no val_loss improvement for {EARLY_PATIENCE} epochs)"
                    )
                    break

        # ── Test metrics every 10 epochs ─────────────────────────────────────
        if (epoch + 1) % 10 == 0:
            test_metrics = evaluate(model, test_loader, num_classes)
            logger.info(f"--- TEST METRICS @ Epoch {epoch+1} ---")
            for k, v in test_metrics.items():
                logger.info(f"  {k}: {v:.4f}")

    # ── Final evaluation với best checkpoint ──────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete.  Best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")
    logger.info(f"Loading best checkpoint: {ckpt_path}")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Checkpoint loaded (saved at epoch {ckpt['epoch']})")

    end_time    = time.time()
    train_time  = end_time - start_time
    logger.info(f"Total Train Time: {train_time:.0f}s  ({train_time/60:.1f} min)")

    flops, params = compute_model_complexity(model)
    logger.info("===== MODEL INFO =====")
    logger.info(f"Model  : {MODEL_NAME}")
    logger.info(f"GFLOPs : {flops:.2f}")
    logger.info(f"Params : {params:.2f}M")

    logger.info("===== FINAL TEST (best checkpoint) =====")
    test_metrics = evaluate(model, test_loader, num_classes)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
