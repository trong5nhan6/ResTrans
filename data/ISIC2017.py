import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

# =========================
# 1. Dataset
# =========================
class ISIC2017(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "imgs")

        csv_file = [f for f in os.listdir(root_dir) if f.endswith(".csv")][0]
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def get_label(self, row):
        melanoma = row["melanoma"]
        sk = row["seborrheic_keratosis"]

        if melanoma == 1: 
            return 0
        elif sk == 1:
            return 1
        else:
            return 2

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row["image_id"]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")

        image = Image.open(img_path).convert("RGB")
        label = self.get_label(row)

        if self.transform:
            image = self.transform(image)

        return image, label