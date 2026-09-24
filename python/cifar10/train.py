"""FP32 training, evaluation, and checkpointing for the CIFAR-10 workload."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

from . import config as C
from .model import Cifar10Net
from .preprocess import build_datasets, fixed_split_loaders


def set_determinism() -> None:
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * x.size(0)
            total += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return correct / total, loss_sum / total


def train_model(device: str = C.DEVICE):
    set_determinism()
    torch.manual_seed(C.SEED)
    train_ds, test_ds = build_datasets()
    train_loader, val_loader, test_loader = fixed_split_loaders(train_ds, test_ds)
    model = Cifar10Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=C.LEARNING_RATE,
                                momentum=C.MOMENTUM, weight_decay=C.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, C.EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        val_acc, val_loss = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "val_acc": val_acc, "val_loss": val_loss})
        print(f"  epoch {epoch:2d}/{C.EPOCHS}  val_acc={val_acc*100:6.2f}%  val_loss={val_loss:.4f}")

    test_acc, test_loss = evaluate(model, test_loader, device)
    return model, {"history": history, "val_acc": history[-1]["val_acc"],
                   "test_acc": test_acc, "test_loss": test_loss}


def save_checkpoint(model, metrics, ckpt_path, meta_path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)
    sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    meta = {
        "experiment": "Step 6.2 — CIFAR-10 cuda-convnet FP32 reference",
        "seed": C.SEED,
        "architecture": {
            "layers": ["conv1(3->32,5x5 valid)", "maxpool(2x2/2)",
                       "conv2(32->32,5x5 valid)", "maxpool(2x2/2)",
                       "conv3(32->64,5x5 valid)", "fc(64->10)"],
            "total_params": C.TOTAL_PARAMS, "total_macs": C.TOTAL_MACS,
        },
        "training": {"optimizer": C.OPTIMIZER, "learning_rate": C.LEARNING_RATE,
                     "momentum": C.MOMENTUM, "batch_size": C.BATCH_SIZE,
                     "epochs": C.EPOCHS, "weight_decay": C.WEIGHT_DECAY},
        "data": {"dataset": "CIFAR-10", "preprocess": "uint8->[0,1] (ToTensor), no mean/std",
                 "split": {"train": [C.TRAIN_START, C.TRAIN_END],
                           "val": [C.VAL_START, C.VAL_END], "test": [0, C.TEST_N]}},
        "metrics": {"val_acc": metrics["val_acc"], "test_acc": metrics["test_acc"],
                    "history": metrics["history"]},
        "checkpoint": {"path": str(ckpt_path), "sha256": sha},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "numpy": np.__version__},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return ckpt_path, sha


def load_checkpoint(ckpt_path, device=C.DEVICE):
    model = Cifar10Net().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model
