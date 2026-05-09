import os
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T


class ISIC2018(Dataset):
    CLASS_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.img_dir   = os.path.join(root_dir, "imgs")
        self.transform = transform

        # ── Locate GroundTruth CSV ──────────────────────────────────────────
        csv_files = [
            f for f in os.listdir(root_dir)
            if f.endswith(".csv") and "GroundTruth" in f
        ]
        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No GroundTruth CSV found in {root_dir}"
            )
        if len(csv_files) > 1:
            raise ValueError(
                f"Multiple GroundTruth CSV files found in {root_dir}: {csv_files}"
            )

        csv_path = os.path.join(root_dir, csv_files[0])
        print(f"Using CSV: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # ── Validate required columns ───────────────────────────────────────
        missing_cols = [c for c in self.CLASS_COLS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV is missing required class columns: {missing_cols}\n"
                f"Found columns: {list(self.df.columns)}"
            )
        if "image" not in self.df.columns:
            raise ValueError(
                f"CSV is missing required 'image' column. "
                f"Found columns: {list(self.df.columns)}"
            )

        # ── Precompute labels once (numpy array, shape [N]) ────────────────
        # __getitem__ uses self.labels[idx] directly — avoids re-parsing CSV rows
        self.labels = (
            self.df[self.CLASS_COLS].values.astype(int).argmax(axis=1)
        )

    # ── Class name helper ─────────────────────────────────────────────────────
    @property
    def class_names(self):
        return self.CLASS_COLS

    # ── Dataset length ────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.df)

    # ── Single sample ─────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img_id = row["image"]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")

        # Informative error when an image is missing or corrupted
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"[ISIC2018] Image not found at index {idx}: {img_path}"
            )
        except (UnidentifiedImageError, OSError) as e:
            raise OSError(
                f"[ISIC2018] Cannot open image at index {idx}: {img_path}\n"
                f"Original error: {e}"
            )

        # Use precomputed label — no redundant CSV parsing at runtime
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label