import torch

DATA_ROOT = "./datasets/ISIC 2018" 
LOG_DIR = "logs_vitb16_resattn"
MODEL_DIR = "models"

SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4

# Per-group LR for vitb16_resattn
LR_ATTN_RES_NORM = 1e-3
LR_MLP_RES_NORM  = 1e-3
LR_HEAD          = 1e-3

# Freeze backbone phase: train only attn_res_norm, mlp_res_norm, head for N epochs first
FREEZE_BACKBONE_EPOCHS = 10  # set 0 để tắt

FOCAL_LOSS = True
GAMMA = 2.0
ALPHA = None
REDUCTION = "mean"

NUM_EPOCHS = 100

USE_CUTMIX = True
USE_MIXUP = True
ALPHA_MIXUP = 0.4
ALPHA_CUTMIX = 1.0
MINORITY_CLASS = None

BLOCK_SIZE = 4  # dùng cho vitb16_resattn

MODEL_NAME = "conv_resattn" # vit_moe "swinv2", "vit", "resnet152", dinov2 "convnext" , resattn, block_resattn, vit_moe vitb16_resattn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"