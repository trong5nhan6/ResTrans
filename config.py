import torch

DATA_ROOT = "./datasets/ISIC 2018" 
LOG_DIR = "logs_vitb16_resattn"
MODEL_DIR = "models"

BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
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

MODEL_NAME = "conv_resattn" # vit_moe "swinv2", "vit", "resnet152", dinov2 "convnext" , resattn, block_resattn, vit_moe vitb16_resattn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"