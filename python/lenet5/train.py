"""L1 FP32 training, evaluation, and reproducible checkpointing (Step 1B).

Trains the locked LeNet-5 model on MNIST (zero-pad-2 -> 32x32), reports
val/test top-1 accuracy, and writes a SHA-256-pinned checkpoint + JSON metadata.

Reproducibility guarantees:
  * fixed seed (``config.SEED``),
  * single-threaded CPU (deterministic floating-point reduction order),
  * ``torch.use_deterministic_algorithms(True)``,
  * explicit weight init (``model.init_weights``),
  * fixed 50k/10k/10k data split (no random split),
  * no LR scheduler, no data augmentation.

The FP32 (L1) path is independent of the future INT8 (L2) reference — nothing
here quantizes.
"""
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
from .model import LeNet5
from .preprocess import build_datasets, fixed_split_loaders


def set_determinism() -> None:
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    torch.manual_seed(C.SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def evaluate(model: nn.Module, loader, device: str) -> tuple[float, float]:
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


def train_model(device: str = C.DEVICE) -> tuple[LeNet5, dict]:
    set_determinism()
    torch.manual_seed(C.SEED)

    train_ds, test_ds = build_datasets()
    train_loader, val_loader, test_loader = fixed_split_loaders(train_ds, test_ds)

    model = LeNet5().to(device)
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
        print(f"  epoch {epoch:2d}/{C.EPOCHS}  val_acc={val_acc*100:6.3f}%  "
              f"val_loss={val_loss:.6f}")

    test_acc, test_loss = evaluate(model, test_loader, device)
    metrics = {
        "history": history,
        "val_acc": history[-1]["val_acc"],
        "val_loss": history[-1]["val_loss"],
        "test_acc": test_acc,
        "test_loss": test_loss,
    }
    return model, metrics


def canonical_state_dict_sha256(state_dict: dict) -> str:
    """Deterministic hash over raw parameter bytes (sorted keys, float32)."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().contiguous().numpy()
        h.update(k.encode("utf-8"))
        h.update(t.tobytes())
    return h.hexdigest()


def save_checkpoint(model: LeNet5, metrics: dict, ckpt_path: Path,
                    meta_path: Path) -> tuple[Path, str]:
    """Write the checkpoint + metadata; return (ckpt_path, file_sha256)."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    # Legacy pickle-only serialization is byte-deterministic (the default
    # zipfile format embeds a timestamp), so the checkpoint SHA-256 is
    # reproducible given deterministic training.
    torch.save(model.state_dict(), ckpt_path, _use_new_zipfile_serialization=False)
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    meta = {
        "experiment": "Step 1B — L1 FP32 LeNet-5 reference",
        "spec": "docs/LENET5_SPEC.md",
        "seed": C.SEED,
        "architecture": {
            "input": [1, 1, C.IMG_H, C.IMG_W],
            "layers": [
                "conv1(1->6, 5x5 valid)", "maxpool(2x2/2)",
                "conv3(6->16, 5x5 valid, fully-connected)", "maxpool(2x2/2)",
                "conv5(16->120, 5x5 valid)", "fc6(120->84)", "fc_out(84->10)",
            ],
            "activations": "ReLU after each conv/FC except output; softmax on output",
            "padding": "valid (no padding); input zero-pad-2 from 28x28 MNIST",
            "total_params": C.TOTAL_PARAMS,
            "total_macs": C.TOTAL_MACS,
        },
        "training": {
            "optimizer": C.OPTIMIZER,
            "learning_rate": C.LEARNING_RATE,
            "momentum": C.MOMENTUM,
            "weight_decay": C.WEIGHT_DECAY,
            "batch_size": C.BATCH_SIZE,
            "epochs": C.EPOCHS,
            "loss": C.LOSS,
            "lr_scheduler": C.LR_SCHEDULER,
            "weight_init": ("kaiming_uniform_(a=sqrt(5)); "
                            "bias uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))"),
            "device": C.DEVICE,
            "determinism": ("seed-pinned; single-thread CPU; "
                            "use_deterministic_algorithms(True)"),
        },
        "data": {
            "dataset": "MNIST (torchvision 0.16.2)",
            "preprocess": "uint8->[0,1] (ToTensor) -> zero-pad-2 -> 32x32; no mean/std",
            "split": {
                "train": [C.TRAIN_START, C.TRAIN_END],
                "val": [C.VAL_START, C.VAL_END],
                "test": [0, C.TEST_N],
            },
        },
        "metrics": {
            "val_acc": metrics["val_acc"],
            "val_loss": metrics["val_loss"],
            "test_acc": metrics["test_acc"],
            "test_loss": metrics["test_loss"],
            "history": metrics["history"],
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": ckpt_sha,
            "canonical_weights_sha256": canonical_state_dict_sha256(model.state_dict()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return ckpt_path, ckpt_sha


def load_checkpoint(ckpt_path: Path, device: str = C.DEVICE) -> LeNet5:
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
