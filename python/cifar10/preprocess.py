"""CIFAR-10 preprocessing and fixed train/val/test split (Step 6.2).

Input: 32x32x3 RGB, uint8 [0,255] -> float [0,1] via ToTensor. No mean/std
normalisation, no augmentation (reproducibility, consistent with the LeNet-5
preprocessing).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config as C

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])  # uint8 -> float [0,1]


def build_datasets(root=None):
    root = Path(root) if root is not None else REPO_ROOT / "data" / "raw"
    tr = make_transform()
    train_ds = datasets.CIFAR10(root=str(root), train=True, download=True, transform=tr)
    test_ds = datasets.CIFAR10(root=str(root), train=False, download=True, transform=tr)
    return train_ds, test_ds


def fixed_split_loaders(train_ds, test_ds):
    gen = torch.Generator()
    gen.manual_seed(C.SEED)
    train = DataLoader(Subset(train_ds, range(C.TRAIN_START, C.TRAIN_END)),
                       batch_size=C.BATCH_SIZE, shuffle=True, num_workers=0, generator=gen)
    val = DataLoader(Subset(train_ds, range(C.VAL_START, C.VAL_END)),
                     batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    test = DataLoader(Subset(test_ds, range(C.TEST_N)),
                      batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    return train, val, test
