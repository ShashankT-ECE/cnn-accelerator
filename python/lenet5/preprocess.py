"""Deterministic MNIST preprocessing for the locked LeNet-5 workload.

Input path (``docs/LENET5_SPEC.md`` §4): MNIST 28x28 uint8 -> float [0,1]
(via ``ToTensor``) -> deterministic zero-pad-2 on all sides -> 32x32. No
mean/std normalisation, no resampling.

Also provides fixed train/val/test split loaders (``config``), all
deterministic — fixed contiguous index ranges, no random split.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from . import config as C

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_transform() -> transforms.Compose:
    """MNIST 28x28 -> [0,1] float -> zero-pad-2 -> 32x32."""
    return transforms.Compose([
        transforms.ToTensor(),   # uint8 -> float32 [0,1], shape (1,28,28)
        transforms.Pad(2),       # constant zero-pad 2 on all sides -> (1,32,32)
    ])


def build_datasets(root: str | Path | None = None):
    """Return (train_ds, test_ds) with the zero-pad-2 transform applied."""
    root = Path(root) if root is not None else REPO_ROOT / "data" / "raw"
    tr = make_transform()
    train_ds = datasets.MNIST(root=str(root), train=True, download=True, transform=tr)
    test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=tr)
    return train_ds, test_ds


def fixed_split_loaders(train_ds, test_ds):
    """Fixed-index train/val/test loaders (deterministic; no random split).

    train = train[0:50000], val = train[50000:60000], test = test[0:10000].
    ``shuffle=True`` on the train loader uses a seed-pinned generator.
    """
    gen = torch.Generator()
    gen.manual_seed(C.SEED)
    train = DataLoader(
        Subset(train_ds, range(C.TRAIN_START, C.TRAIN_END)),
        batch_size=C.BATCH_SIZE, shuffle=True, num_workers=0, generator=gen)
    val = DataLoader(
        Subset(train_ds, range(C.VAL_START, C.VAL_END)),
        batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    test = DataLoader(
        Subset(test_ds, range(C.TEST_N)),
        batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)
    return train, val, test
