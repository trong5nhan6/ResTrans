import torch

# =========================
# Paths
# =========================
DATA_ROOT  = "./datasets/ISIC 2018"
LOG_DIR    = "logs_vitb16_resattn"
MODEL_DIR  = "models"

# =========================
# Reproducibility
# =========================
SEED = 42

# =========================
# DataLoader
# =========================
BATCH_SIZE  = 32
NUM_WORKERS = 4

# =========================
# Optimizer
# =========================
LR            = 1e-4
WEIGHT_DECAY  = 1e-4

# Layer-wise LR decay — chỉ áp dụng cho vitb16_resattn
USE_LLRD   = True
LLRD_DECAY = 0.75   # lr_layer_i = LR * LLRD_DECAY^(num_layers - 1 - i)

# =========================
# Scheduler
# =========================
NUM_EPOCHS     = 100
WARMUP_EPOCHS  = 5       # Linear warmup trước khi chuyển sang CosineAnnealing

# =========================
# Gradient clipping
# =========================
GRADIENT_CLIP = 1.0      # max_norm; đặt 0 để tắt

# =========================
# Loss
# =========================
FOCAL_LOSS  = False      # True = FocalLoss, False = CrossEntropy + class weights
GAMMA       = 2.0
ALPHA       = None
REDUCTION   = "mean"

# =========================
# Augmentation / Regularisation
# =========================
USE_CUTMIX    = True
USE_MIXUP     = True
ALPHA_MIXUP   = 0.4
ALPHA_CUTMIX  = 1.0
MINORITY_CLASS = None

# =========================
# Checkpoint / Early-stop
# =========================
SAVE_BEST       = True    # lưu model tốt nhất theo val_acc
EARLY_STOP      = True    # dừng sớm nếu val_loss không cải thiện
EARLY_PATIENCE  = 15      # số epoch chờ trước khi dừng

# =========================
# Model
# =========================
MODEL_NAME = "vitb16_resattn"  # focal point của project
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
