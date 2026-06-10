"""
COD10K Dataset Loader

Dataset directory structure expected:
    cod10k/
        TrainDataset/
            Imgs/          # RGB images  (.jpg)
            GT/            # Binary masks (.png, 0=bg, 255=fg)
        TestDataset/
            COD10K/
                Imgs/
                GT/

Also supports CAMO and NC4K with the same structure convention.
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Preprocessing Strategy
# ---------------------------------------------------------------------------
# USEFUL for COD:
#   1. Normalization (ImageNet stats) — required; backbone expects it.
#   2. CLAHE (Contrast Limited AHE) — helpful for very dark/low-contrast images.
#      Locally enhances contrast without blowing out highlights.
#      Use ONLY in augmentation pipeline (not deterministic) to avoid
#      overfitting to enhanced images.
#   3. Random brightness/contrast — forces model to be illumination-invariant.
#
# NOT useful / harmful:
#   4. Global histogram equalization — global, destroys color relationships.
#   5. Sharpening filters — adds artificial edges not in ground truth.
#   6. Color jitter (strong) — camouflage is color-based; extreme jitter
#      destroys the very signal the model must learn.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.75, 1.33), p=1.0),

        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),

        # Color/Contrast — subtle is KEY for COD
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
        ], p=0.5),

        # CLAHE for low-contrast images
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # Noise / Blur (simulate real-world capture artifacts)
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),

        # Normalize + ToTensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 384) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CODDataset(Dataset):
    """
    Loads image + binary mask pairs.
    Works for COD10K, CAMO, and NC4K given correct paths.
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        img_size: int = 384,
    ):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform or get_val_transforms(img_size)

        # Collect paired samples
        self.samples = self._collect_samples()
        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No image-mask pairs found.\n  img_dir: {img_dir}\n  mask_dir: {mask_dir}"
            )

    def _collect_samples(self) -> list:
        samples = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            stem = img_path.stem
            # COD10K masks have same stem as image
            mask_path = self.mask_dir / f"{stem}.png"
            if not mask_path.exists():
                # Try .jpg mask (some datasets)
                mask_path = self.mask_dir / f"{stem}.jpg"
            if mask_path.exists():
                samples.append((img_path, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load image (BGR → RGB)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask — binary {0, 255} → {0, 1}
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        # Generate edge map from mask (for edge supervision)
        edge = self._generate_edge(mask)

        if self.transform:
            # albumentations needs H×W×C image and H×W mask
            transformed = self.transform(image=img, masks=[mask, edge])
            img  = transformed["image"]          # C×H×W tensor
            mask = transformed["masks"][0]       # H×W tensor
            edge = transformed["masks"][1]

        # Ensure mask and edge have channel dim
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if edge.ndim == 2:
            edge = edge.unsqueeze(0)

        return img, mask, edge

    @staticmethod
    def _generate_edge(mask: np.ndarray, dilation: int = 2) -> np.ndarray:
        """Morphological edge from binary mask."""
        mask_u8 = (mask * 255).astype(np.uint8)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation * 2 + 1,) * 2)
        dilated = cv2.dilate(mask_u8, kernel)
        eroded  = cv2.erode(mask_u8, kernel)
        edge    = ((dilated - eroded) > 0).astype(np.float32)
        return edge


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    img_size: int = 384,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:

    train_ds = CODDataset(
        train_img_dir, train_mask_dir,
        transform=get_train_transforms(img_size),
        img_size=img_size,
    )
    val_ds = CODDataset(
        val_img_dir, val_mask_dir,
        transform=get_val_transforms(img_size),
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Multi-dataset loader (COD10K + CAMO + NC4K combined)
# ---------------------------------------------------------------------------

class CombinedCODDataset(Dataset):
    """
    Merges multiple COD datasets into one.
    Useful for cross-dataset generalization (improvement #3).
    """

    def __init__(self, dataset_configs: list, transform: Optional[Callable] = None):
        """
        dataset_configs: list of (img_dir, mask_dir) tuples
        """
        self.datasets = [
            CODDataset(img_dir, mask_dir, transform=transform)
            for img_dir, mask_dir in dataset_configs
        ]
        self.lengths = [len(d) for d in self.datasets]
        self.total   = sum(self.lengths)
        self.offsets = [0] + list(np.cumsum(self.lengths[:-1]))

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        for ds, offset in zip(self.datasets, self.offsets):
            if idx < offset + len(ds):
                return ds[idx - offset]
        raise IndexError(f"Index {idx} out of range for CombinedCODDataset")
