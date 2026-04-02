import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

class ISIC2018(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "imgs")

        # tìm file CSV có chữ GroundTruth
        csv_files = [
            f for f in os.listdir(root_dir)
            if f.endswith(".csv") and "GroundTruth" in f
        ]

        if len(csv_files) == 0:
            raise FileNotFoundError("No GroundTruth CSV found!")
        elif len(csv_files) > 1:
            raise ValueError(f"Multiple GroundTruth CSV found: {csv_files}")

        csv_path = os.path.join(root_dir, csv_files[0])
        print(f"Using CSV: {csv_path}")

        self.df = pd.read_csv(csv_path)

        self.transform = transform
        self.class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
        self.labels = self.df[self.class_cols].values.astype(int).argmax(axis=1)

    def __len__(self):
        return len(self.df)

    def get_label(self, row):
        labels = row[self.class_cols].values.astype(int)
        return labels.argmax()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row["image"]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")

        image = Image.open(img_path).convert("RGB")
        label = self.get_label(row)

        if self.transform:
            image = self.transform(image)

        return image, label