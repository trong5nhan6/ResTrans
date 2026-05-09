import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from thop import profile
import logging
import os
import torchvision.transforms as T
import torch.nn as nn
import numpy as np


# =========================
# Model complexity
# =========================
def compute_model_complexity(model, input_size=(1, 3, 224, 224), device="cuda"):
    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    params = sum(p.numel() for p in model.parameters())
    try:
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except Exception:
        flops = 0
    return flops / 1e9, params / 1e6   # GFLOPs, Millions


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =========================
# Accuracy helpers
# =========================
def accuracy_topk(logits, labels, topk=(1,)):
    maxk = max(topk)
    batch_size = labels.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size).item())
    return res


# =========================
# Metrics (with per-class breakdown)
# =========================
def compute_metrics(logits, labels, num_classes):
    """
    logits : [N, C]  (raw logits, not softmax)
    labels : [N]
    """
    probs    = F.softmax(logits, dim=1).detach().cpu().numpy()
    preds    = logits.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    acc1, acc5 = accuracy_topk(logits, labels, topk=(1, min(5, num_classes)))

    precision = precision_score(labels_np, preds, average="macro", zero_division=0)
    recall    = recall_score(   labels_np, preds, average="macro", zero_division=0)
    f1        = f1_score(       labels_np, preds, average="macro", zero_division=0)

    # Per-class F1 (useful for debugging minority classes)
    per_class_f1 = f1_score(labels_np, preds, average=None, zero_division=0)

    try:
        roc_auc = roc_auc_score(
            labels_np, probs,
            multi_class="ovr", average="macro"
        )
    except Exception:
        roc_auc = 0.0

    metrics = {
        "Acc@1":     acc1,
        "Acc@5":     acc5,
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
        "Macro F1":  f1,
        "ROC-AUC":   roc_auc,
    }

    # Attach per-class F1 as separate keys
    for i, v in enumerate(per_class_f1):
        metrics[f"F1_class{i}"] = float(v)

    return metrics


# =========================
# Logger
# =========================
def setup_logger(log_dir="logs", log_name="train.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    # Dùng tên logger riêng cho mỗi file log → tránh collision khi chạy nhiều lần
    logger_name = f"ISIC_{log_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def log_dataset_info(logger, dataset, name="Train"):
    logger.info(f"===== {name.upper()} DATASET =====")
    logger.info(f"Total samples: {len(dataset)}")
    labels   = np.array(dataset.labels)
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution:")
    for u, c in zip(unique, counts):
        logger.info(f"  Class {u}: {c} samples ({c/len(labels):.2%})")


# =========================
# Augmentation
# =========================
def get_transform(is_train=True):
    """
    Dermoscopy-aware augmentation.

    Key changes vs original:
    - RandomResizedCrop   : simulates lesion at different scales / off-center crops
    - RandomRotation(180) : dermoscopy images have NO canonical orientation → full rotation
    - Stronger ColorJitter: different dermoscope devices produce different colour casts
    - Val/test resize to 256 → CenterCrop 224: standard ViT evaluation protocol
    """
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(180),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.02),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])


# =========================
# Loss functions
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha      # tensor [C] hoặc None
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, reduction="none")
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            at         = self.alpha.to(inputs.device).gather(0, targets)
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# =========================
# Mixup
# =========================
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index   = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data_class_aware(x, y, alpha=0.4, minority_classes=None):
    batch_size = x.size(0)
    lam        = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    if minority_classes is None or len(minority_classes) == 0:
        index = torch.randperm(batch_size).to(x.device)
    else:
        minority_idx = torch.cat(
            [(y == cls).nonzero(as_tuple=True)[0] for cls in minority_classes]
        )
        if len(minority_idx) == 0:
            index = torch.randperm(batch_size).to(x.device)
        else:
            index_min  = minority_idx[torch.randint(len(minority_idx), (batch_size,))].to(x.device)
            other_idx  = torch.randperm(batch_size).to(x.device)
            mask       = torch.rand(batch_size).to(x.device) < 0.5
            index      = torch.where(mask, index_min, other_idx)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


# =========================
# CutMix helpers
# =========================
def rand_bbox(size, lam):
    """
    size : (B, C, H, W)
    Returns bounding box (y1, x1, y2, x2) in image coords [H-axis, W-axis].
    """
    H, W   = size[2], size[3]
    cut_h  = int(H * np.sqrt(1.0 - lam))
    cut_w  = int(W * np.sqrt(1.0 - lam))

    # centre
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))

    return y1, x1, y2, x2


def cutmix_data(x, y, alpha=1.0):
    """
    Bug fixes vs original:
    1. x.clone() → không sửa tensor gốc in-place.
    2. rand_bbox trả về (y1,x1,y2,x2) — đúng thứ tự [B,C,H,W].
    """
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    x = x.clone()                              # ← FIX: không sửa batch gốc
    y1, x1, y2, x2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Lambda thực tế dựa trên diện tích vùng cắt
    lam = 1 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
    return x, y, y[index], lam


def cutmix_data_class_aware(x, y, alpha=1.0, minority_classes=None):
    batch_size = x.size(0)
    lam        = np.random.beta(alpha, alpha)
    index      = torch.randperm(batch_size).to(x.device)

    if minority_classes is not None and len(minority_classes) > 0:
        minority_idx = torch.cat(
            [(y == cls).nonzero(as_tuple=True)[0] for cls in minority_classes]
        )
        if len(minority_idx) > 0:
            index_min = minority_idx[torch.randint(len(minority_idx), (batch_size,))].to(x.device)
            mask      = torch.rand(batch_size).to(x.device) < 0.5
            index     = torch.where(mask, index_min, index)

    x = x.clone()                              # ← FIX: không sửa batch gốc
    y1, x1, y2, x2 = rand_bbox(x.size(), lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3)))
    return x, y, y[index], lam
