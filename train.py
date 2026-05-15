import torch
import torch.nn as nn
from utils.utils import setup_logger, log_dataset_info, compute_model_complexity, compute_metrics, get_transform, FocalLoss, cutmix_data, mixup_data, mixup_criterion, cutmix_data_class_aware, mixup_data_class_aware, count_params
import os
import random
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from config import DATA_ROOT, LOG_DIR, MODEL_DIR, BATCH_SIZE, NUM_WORKERS, LR, WEIGHT_DECAY, DEVICE, NUM_EPOCHS, MODEL_NAME, USE_CUTMIX, USE_MIXUP, ALPHA_MIXUP, ALPHA_CUTMIX, MINORITY_CLASS, FOCAL_LOSS, GAMMA, ALPHA, REDUCTION, LR_ATTN_RES_NORM, LR_MLP_RES_NORM, LR_HEAD, FREEZE_BACKBONE_EPOCHS, SEED
from data.ISIC2018 import ISIC2018
from model.basemodel import BaseModel
from model.block_resattn import BlockAttnResClassifier
from model.resattn import FullAttnResClassifier
from model.vit_moe import ViT_BlockMoE
from model.vitb16_resattn import ViTB16_AttnRes
from model.conv_resattn import ConvNeXt_AttnRes
from tqdm import tqdm
import numpy as np
import time

# =========================
# 8. Train / Eval
# =========================
def train_one_epoch(model, train_loader, val_loader,
                    use_mixup=USE_MIXUP, use_cutmix=USE_CUTMIX,
                    alpha_mixup=ALPHA_MIXUP, alpha_cutmix=ALPHA_CUTMIX,
                    minority_class=MINORITY_CLASS):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # ===== Randomly apply Class-aware Mixup or CutMix =====
        r = np.random.rand()
        if USE_MIXUP and r < 0.5:
            if minority_class is None:
                # Mixup thường
                imgs, targets_a, targets_b, lam = mixup_data(
                    imgs, labels, alpha=alpha_mixup
                )   
            else:
                # Mixup class-aware
                imgs, targets_a, targets_b, lam = mixup_data_class_aware(
                    imgs, labels, alpha=alpha_mixup, minority_classes=minority_class
                )

        elif USE_CUTMIX:
            if minority_class is None:
                # CutMix thường
                imgs, targets_a, targets_b, lam = cutmix_data(
                    imgs, labels, alpha=alpha_cutmix
                )
            else:
                # CutMix class-aware
                imgs, targets_a, targets_b, lam = cutmix_data_class_aware(
                    imgs, labels, alpha=alpha_cutmix, minority_classes=minority_class
                )
        else:
            targets_a, targets_b, lam = labels, labels, 1.0
        outputs = model(imgs)

        # ===== LOSS =====
        if MODEL_NAME == "vit_moe":
            final_logits, block_logits = outputs

            if lam < 1.0:
                # ----- Mixup -----
                main_loss = mixup_criterion(criterion, final_logits, targets_a, targets_b, lam)

                aux_loss = sum(
                    mixup_criterion(criterion, logit, targets_a, targets_b, lam)
                    for logit in block_logits
                ) / len(block_logits)

            else:
                # ----- Normal -----
                main_loss = criterion(final_logits, labels)

                aux_loss = sum(
                    criterion(logit, labels)
                    for logit in block_logits
                ) / len(block_logits)

            loss = main_loss + 0.3 * aux_loss

        else:
            # ===== Model thường =====
            if lam < 1.0:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                final_logits = outputs
            else:
                loss = criterion(outputs, labels)
                final_logits = outputs

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if MODEL_NAME == "vit_moe":
            preds = final_logits.argmax(dim=1)
        else:
            preds = outputs.argmax(dim=1)

        # ===== Train accuracy =====
        if lam < 1.0:
            # weighted correct for Mixup / CutMix
            train_correct += (lam * (preds == targets_a).sum().item() +
                              (1 - lam) * (preds == targets_b).sum().item())
        else:
            train_correct += (preds == labels).sum().item()

        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # ===== Validation =====
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)  

            if MODEL_NAME == "vit_moe":
                final_logits, block_logits = outputs

                main_loss = criterion(final_logits, labels)
                aux_loss = sum(
                    criterion(logit, labels)
                    for logit in block_logits
                ) / len(block_logits)

                loss = main_loss + 0.3 * aux_loss

            else:
                loss = criterion(outputs, labels)
                final_logits = outputs

            val_loss += loss.item()

            preds = final_logits.argmax(dim=1) 
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    return train_loss, train_acc, val_loss, val_acc


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()

    all_logits = []
    all_labels = []

    for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        if MODEL_NAME == 'vit_moe':
            final_logits, block_logits = model(imgs)
            outputs = final_logits
        else:
            outputs = model(imgs)

        all_logits.append(outputs)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_logits, all_labels, num_classes)

    return metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

if __name__ == "__main__":
    set_seed(SEED)

    # =========================
    # 3. Load dataset
    # =========================
    train_dataset = ISIC2018(os.path.join(DATA_ROOT, "train"), transform=get_transform(is_train=True))
    val_dataset   = ISIC2018(os.path.join(DATA_ROOT, "val"), transform=get_transform(is_train=False))
    test_dataset  = ISIC2018(os.path.join(DATA_ROOT, "test"), transform=get_transform(is_train=False))
    print('device:', DEVICE)

    # =========================
    # 4. Handle imbalance
    # =========================
    labels = train_dataset.labels
    class_count = np.bincount(labels)
    num_classes = len(class_count)
    print("Class distribution:", class_count)

    class_weights = 1. / class_count
    sample_weights = [class_weights[l] for l in labels]
    print('sample_weights:', class_weights)
    g = torch.Generator()
    g.manual_seed(SEED)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=g)

    # =========================
    # 5. DataLoader
    # =========================
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS,
                              worker_init_fn=worker_init_fn, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              worker_init_fn=worker_init_fn)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              worker_init_fn=worker_init_fn)

    # =========================
    # 6. Model (baseline)
    # =========================
    if MODEL_NAME == 'resattn':
        model = FullAttnResClassifier(num_classes=num_classes)
    elif MODEL_NAME == 'block_resattn':
        model = BlockAttnResClassifier(num_classes=num_classes)
    elif MODEL_NAME == 'vit_moe':
        model = ViT_BlockMoE(num_classes=num_classes)
    elif MODEL_NAME == 'vitb16_resattn':
        model = ViTB16_AttnRes(block_size=4, num_classes=num_classes)
    elif MODEL_NAME == 'conv_resattn':
        model = ConvNeXt_AttnRes(num_classes=num_classes)
    else:
        model = BaseModel(model_name=MODEL_NAME, num_classes=num_classes)

    model = model.to(DEVICE)

    # =========================
    # 7. Loss + Optimizer
    # =========================
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    if FOCAL_LOSS:
        criterion = FocalLoss(gamma=GAMMA, alpha=ALPHA, reduction=REDUCTION)
    else:
        criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # criterion = FocalLoss(num_classes=num_classes, gamma=1.0, weight=class_weights_tensor)
    _TRAINABLE_FREEZE = ('attn_res_norm', 'mlp_res_norm', 'head')

    def _make_full_optimizer():
        pg = model.get_param_groups(
            lr_base=LR,
            lr_attn_res_norm=LR_ATTN_RES_NORM,
            lr_mlp_res_norm=LR_MLP_RES_NORM,
            lr_head=LR_HEAD,
        )
        return torch.optim.AdamW(pg, weight_decay=WEIGHT_DECAY)

    if MODEL_NAME == 'vitb16_resattn' and FREEZE_BACKBONE_EPOCHS > 0:
        for name, p in model.named_parameters():
            p.requires_grad = any(k in name for k in _TRAINABLE_FREEZE)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=FREEZE_BACKBONE_EPOCHS, eta_min=1e-6)
    elif MODEL_NAME == 'vitb16_resattn':
        optimizer = _make_full_optimizer()
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    # =========================
    # 9. Training loop
    # =========================
    log_name = f"{MODEL_NAME}_mixup_{USE_MIXUP}_cutmix_{USE_CUTMIX}_minority_{MINORITY_CLASS}_FOCALLOSS_{FOCAL_LOSS}_zero_init.log"
    logger = setup_logger(log_dir=LOG_DIR, log_name=log_name)

    log_dataset_info(logger, train_dataset, "Train")
    log_dataset_info(logger, val_dataset, "Validation")
    log_dataset_info(logger, test_dataset, "Test")
    print("Total parameters:", count_params(model))

    EPOCHS = NUM_EPOCHS
    start_time = time.time()
    for epoch in range(EPOCHS):
        # unfreeze backbone sau freeze phase
        if MODEL_NAME == 'vitb16_resattn' and FREEZE_BACKBONE_EPOCHS > 0 and epoch == FREEZE_BACKBONE_EPOCHS:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = _make_full_optimizer()
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - FREEZE_BACKBONE_EPOCHS, eta_min=1e-6)
            logger.info(f"Epoch {epoch+1}: Unfreeze backbone — chuyển sang full param groups với lr riêng.")

        train_loss, train_acc, val_loss, val_acc = train_one_epoch(
            model, train_loader, val_loader,
            use_mixup=USE_MIXUP,   
            use_cutmix=USE_CUTMIX,
            alpha_mixup=ALPHA_MIXUP,
            alpha_cutmix=ALPHA_CUTMIX,
            minority_class=MINORITY_CLASS
        )
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}]")
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        scheduler.step()

        # ===== Test every 20 epochs =====
        if (epoch + 1) % 10 == 0:
            test_metrics = evaluate(model, test_loader, num_classes)
            logger.info(f"--- TEST METRICS @ Epoch {epoch+1} ---")
            for k, v in test_metrics.items():
                logger.info(f"{k}: {v:.4f}")
                
    end_time = time.time()
    train_time = end_time - start_time
    logger.info(f"Total Train Time: {train_time:.2f}s")

    flops, params = compute_model_complexity(model)
    logger.info("===== MODEL INFO =====")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"GFLOPs: {flops:.2f}")
    logger.info(f"Params (M): {params:.2f}")
    test_metrics = evaluate(model, test_loader, num_classes)