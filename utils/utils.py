import torch
import torch.nn.functional as F
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from thop import profile
import logging
import os
import torchvision.transforms as T
import torch.nn as nn
import numpy as np

def compute_model_complexity(model, input_size=(1, 3, 224, 224), device="cuda"):
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    # convert
    flops = flops / 1e9      # GFLOPs
    params = params / 1e6    # Millions

    return flops, params

# =========================
# 1. Accuracy
# =========================
def accuracy_topk(logits, labels, topk=(1,)):
    maxk = max(topk)
    batch_size = labels.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [K, B]

    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size).item())

    return res  # list


# =========================
# 2. All metrics
# =========================
def compute_metrics(logits, labels, num_classes):
    """
    logits: [N, C]
    labels: [N]
    """

    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Accuracy
    acc1, acc5 = accuracy_topk(logits, labels, topk=(1, min(5, num_classes)))

    # Precision / Recall / F1
    precision = precision_score(labels_np, preds, average="macro", zero_division=0)
    recall = recall_score(labels_np, preds, average="macro", zero_division=0)
    f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

    # Macro F1 (same as above nhưng để rõ)
    macro_f1 = f1

    # ROC-AUC (multi-class)
    try:
        roc_auc = roc_auc_score(
            labels_np,
            probs,
            multi_class="ovr",
            average="macro"
        )
    except:
        roc_auc = 0.0  # tránh crash khi thiếu class trong batch

    return {
        "Acc@1": acc1,
        "Acc@5": acc5,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Macro F1": macro_f1,
        "ROC-AUC": roc_auc,
    }


def setup_logger(log_dir="logs", log_name="train.log"):
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger("ISIC_Logger")
    logger.setLevel(logging.INFO)

    # tránh duplicate handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_dataset_info(logger, dataset, name="Train"):
    import numpy as np

    logger.info(f"===== {name.upper()} DATASET =====")
    logger.info(f"Total samples: {len(dataset)}")

    # dùng labels đã lưu sẵn
    labels = np.array(dataset.labels)

    # class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution:")
    for u, c in zip(unique, counts):
        logger.info(f"  Class {u}: {c} samples ({c/len(labels):.2%})")


def get_transform(is_train=True):
    if is_train:
        return T.Compose([
            T.Resize((224,224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # nhẹ hơn
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, weight=None, device="cuda"):
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.num_classes = num_classes
        self.weight = weight.to(device) if weight is not None else None
        
        # Learnable class bias term
        self.class_bias = nn.Parameter(torch.zeros(num_classes, device=device))

    def forward(self, logits, targets):
        # Add learnable bias
        logits = logits + self.class_bias

        # Compute CE loss without reduction
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')

        # Focal scaling
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma * ce_loss)

        return loss.mean()

class Cutout:
    def __init__(self, n_holes=1, length=50):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img: C x H x W tensor
        h, w = img.shape[1], img.shape[2]

        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            y = random.randint(0, h-1)
            x = random.randint(0, w-1)

            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)

            mask[:, y1:y2, x1:x2] = 0.0

        return img * mask

def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda based on area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def mixup_data_class_aware(x, y, alpha=0.4, minority_classes=None):
    """
    Mixup Class-Aware: ưu tiên trộn với sample từ các minority class
    x: (B, C, H, W)
    y: (B,)
    minority_classes: list các class ít dữ liệu, ví dụ [2,3,5]
    """
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    if minority_classes is None or len(minority_classes) == 0:
        # dùng permutation bình thường
        index = torch.randperm(batch_size).to(x.device)
    else:
        # Tạo danh sách index của tất cả ảnh thuộc minority classes
        minority_idx = [ (y == cls).nonzero(as_tuple=True)[0] for cls in minority_classes ]
        minority_idx = torch.cat(minority_idx)
        if len(minority_idx) == 0:
            index_min = torch.randperm(batch_size).to(x.device)
        else:
            index_min = minority_idx[torch.randint(len(minority_idx), (batch_size,))].to(x.device)

        # permutation bình thường
        other_idx = torch.randperm(batch_size).to(x.device)
        mask = torch.rand(batch_size).to(x.device) < 0.5
        index = torch.where(mask, index_min, other_idx)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data_class_aware(x, y, alpha=1.0, minority_classes=None):
    batch_size, C, H, W = x.size()
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    index = torch.randperm(batch_size).to(x.device)

    if minority_classes is not None and len(minority_classes) > 0:
        minority_idx = torch.cat([(y == cls).nonzero(as_tuple=True)[0] for cls in minority_classes])
        if len(minority_idx) > 0:
            index_min = minority_idx[torch.randint(len(minority_idx), (batch_size,))].to(x.device)
            mask = torch.rand(batch_size).to(x.device) < 0.5
            index = torch.where(mask, index_min, index)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
