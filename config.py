import torch

DATA_ROOT = "./datasets/ISIC 2018" 
LOG_DIR = "logs"
MODEL_DIR = "models"

BATCH_SIZE = 64    
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 100

USE_CUTMIX = False
USE_MIXUP = False
ALPHA_MIXUP = 0.4
ALPHA_CUTMIX = 1.0
MINORITY_CLASS = [5, 6, 3]

MODEL_NAME = "convnext" # "swinv2", "vit", "resnet152", "convnext" , resattn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"