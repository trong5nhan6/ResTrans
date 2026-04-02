import torch

DATA_ROOT = "./datasets/ISIC 2018" 
LOG_DIR = "logs"
MODEL_DIR = "models"

BATCH_SIZE = 64    
NUM_WORKERS = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 30

MODEL_NAME = "resattn" # "swinv2", "vit", "resnet152", "convnext" , resattn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"