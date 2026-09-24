#!/usr/bin/env python3
"""V2 Step 2.1b: retrain the CIFAR-10 network (same architecture) -> r2 FP32 checkpoint.

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/model/retrain/train_cifar10_r2.py

Network: legacy ``cifar10.model.Cifar10Net`` (imported, unchanged; no BatchNorm,
no added layers). Input preprocessing: legacy ``cifar10.preprocess.make_transform``
(ToTensor: uint8 -> float [0,1], no mean/std), so the per-tensor INT8 input scale
of the legacy pipeline applies unchanged.

Protocol (honest):
    train = CIFAR-10 train[0:45000]   (augmented: random crop 32 pad 4 + h-flip)
    val   = CIFAR-10 train[45000:50000] (no augmentation) -> epoch selection ONLY
    test  = never touched by this script.

Recipe: SGD momentum 0.9 nesterov, weight decay 5e-4, cosine LR from 0.05 over
EPOCHS epochs (per-epoch step, eta_min 0), batch 128, fixed seeds.
Selected epoch = argmax FP32 val accuracy (earliest on ties).

Outputs:
    v2/results/cifar10_retrain_log.csv       per-epoch train loss / val acc (+ metadata)
    v2/model/retrain/cifar10_fp32_r2.pt      selected state_dict (legacy save format)
    v2/model/retrain/cifar10_fp32_r2.pt.sha256
    v2/model/retrain/cifar10_fp32_r2_meta.json
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402  (puts legacy python/ on sys.path)
from common import REPO_ROOT, RESULTS_DIR, sha256_file  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
from torchvision import datasets  # noqa: E402

from cifar10.model import Cifar10Net  # noqa: E402  (legacy, read-only)
from cifar10.preprocess import make_transform  # noqa: E402  (legacy, read-only)

SEED = 20260924
EPOCHS = 60
BATCH = 128
LR0 = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
PAD = 4
TRAIN_RANGE = (0, 45000)
VAL_RANGE = (45000, 50000)

RETRAIN_DIR = Path(__file__).resolve().parent
CKPT = RETRAIN_DIR / "cifar10_fp32_r2.pt"
META = RETRAIN_DIR / "cifar10_fp32_r2_meta.json"
LOG_CSV = RESULTS_DIR / "cifar10_retrain_log.csv"
LOG_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_acc", "val_correct",
              "val_total", "val_loss", "epoch_time_s", "selected"]


def _rel(p: Path) -> str:
    p = p.resolve()
    return p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else str(p)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def load_split(ds, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ds[lo:hi] through the legacy transform (float32 [0,1])."""
    x = torch.stack([ds[i][0] for i in range(lo, hi)])
    y = torch.tensor([ds[i][1] for i in range(lo, hi)], dtype=torch.int64)
    assert x.dtype == torch.float32 and x.shape[1:] == (3, 32, 32)
    return x, y


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """RandomCrop(32, padding=4, zero fill) + RandomHorizontalFlip(0.5), per sample."""
    n = x.shape[0]
    xp = F.pad(x, (PAD, PAD, PAD, PAD))                              # (N,3,40,40), zeros
    oy = torch.randint(0, 2 * PAD + 1, (n,), generator=gen)
    ox = torch.randint(0, 2 * PAD + 1, (n,), generator=gen)
    flip = torch.rand(n, generator=gen) < 0.5
    ar = torch.arange(32)
    rows = (oy[:, None] + ar)[:, None, :, None]                      # (N,1,32,1)
    cols = ox[:, None] + ar                                          # (N,32)
    cols = torch.where(flip[:, None], cols.flip(1), cols)[:, None, None, :]
    nidx = torch.arange(n)[:, None, None, None]
    cidx = torch.arange(3)[None, :, None, None]
    return xp[nidx, cidx, rows, cols]


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[int, float]:
    model.eval()
    correct, loss_sum = 0, 0.0
    for s in range(0, x.shape[0], 1000):
        logits = model(x[s:s + 1000])
        loss_sum += F.cross_entropy(logits, y[s:s + 1000], reduction="sum").item()
        correct += int((logits.argmax(1) == y[s:s + 1000]).sum())
    return correct, loss_sum / x.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--log", type=Path, default=LOG_CSV)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    args = ap.parse_args()

    set_seeds(SEED)
    root = str(REPO_ROOT / "data" / "raw")
    train_ds = datasets.CIFAR10(root=root, train=True, download=False, transform=make_transform())
    x_tr, y_tr = load_split(train_ds, *TRAIN_RANGE)
    x_va, y_va = load_split(train_ds, *VAL_RANGE)

    model = Cifar10Net()                       # legacy init (seeded)
    opt = torch.optim.SGD(model.parameters(), lr=LR0, momentum=MOMENTUM, nesterov=True,
                          weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)
    gen = torch.Generator().manual_seed(SEED)

    rows, best = [], None
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        te = time.perf_counter()
        model.train()
        lr = opt.param_groups[0]["lr"]
        perm = torch.randperm(x_tr.shape[0], generator=gen)
        loss_sum, correct, seen = 0.0, 0, 0
        for s in range(0, perm.numel(), BATCH):
            idx = perm[s:s + BATCH]
            xb, yb = augment(x_tr[idx], gen), y_tr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * idx.numel()
            correct += int((logits.argmax(1) == yb).sum())
            seen += idx.numel()
        sched.step()
        vc, vl = evaluate(model, x_va, y_va)
        dt = time.perf_counter() - te
        if best is None or vc > best["val_correct"]:
            best = {"epoch": epoch, "val_correct": vc, "state": copy.deepcopy(model.state_dict())}
        row = common.base_meta(net="cifar10", layer="all", source="model",
                               duration_s=round(dt, 3), num_inferences=seen + x_va.shape[0])
        row.update(epoch=epoch, lr=f"{lr:.8g}", train_loss=f"{loss_sum / seen:.6f}",
                   train_acc=f"{100.0 * correct / seen:.2f}", val_acc=f"{100.0 * vc / x_va.shape[0]:.2f}",
                   val_correct=vc, val_total=x_va.shape[0], val_loss=f"{vl:.6f}",
                   epoch_time_s=round(dt, 2), selected="")
        rows.append(row)
        common.write_results_csv(args.log, rows, LOG_FIELDS)
        print(f"epoch {epoch:2d}/{args.epochs} lr={lr:.5f} loss={loss_sum / seen:.4f} "
              f"train={100.0 * correct / seen:.2f}% val={100.0 * vc / x_va.shape[0]:.2f}% "
              f"({dt:.1f}s)", flush=True)
    train_time = time.perf_counter() - t_start

    for r in rows:
        r["selected"] = int(r["epoch"] == best["epoch"])
    common.write_results_csv(args.log, rows, LOG_FIELDS)

    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], args.ckpt, _use_new_zipfile_serialization=False)
    sha = sha256_file(args.ckpt)
    args.ckpt.with_name(args.ckpt.name + ".sha256").write_text(f"{sha}  {args.ckpt.name}\n")
    meta = {
        "experiment": "V2 Step 2.1b - CIFAR-10 retrain r2 (same architecture), FP32",
        "generator": "v2/model/retrain/train_cifar10_r2.py",
        "architecture": "legacy cifar10.model.Cifar10Net (unchanged; no BatchNorm)",
        "preprocess": "legacy cifar10.preprocess.make_transform (ToTensor, [0,1], no mean/std)",
        "augmentation": "RandomCrop(32, padding=4, zero fill) + horizontal flip p=0.5 (train only)",
        "split": {"train": list(TRAIN_RANGE), "val": list(VAL_RANGE),
                  "test": "not used by this script"},
        "training": {"optimizer": "SGD", "momentum": MOMENTUM, "nesterov": True,
                     "weight_decay": WEIGHT_DECAY, "lr0": LR0,
                     "lr_schedule": f"CosineAnnealingLR(T_max={args.epochs}, eta_min=0), per epoch",
                     "batch_size": BATCH, "epochs_run": args.epochs, "seed": SEED,
                     "seeds": "random, numpy, torch.manual_seed and data generator = SEED"},
        "selection": "argmax FP32 val accuracy over epochs (earliest on ties); test not used",
        "selected_epoch": best["epoch"],
        "val_correct": best["val_correct"], "val_total": int(x_va.shape[0]),
        "val_acc_pct": round(100.0 * best["val_correct"] / x_va.shape[0], 2),
        "train_time_s": round(train_time, 1),
        "torch_threads": torch.get_num_threads(),
        "checkpoint": {"path": _rel(args.ckpt), "sha256": sha},
        "log_csv": _rel(args.log),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "torchvision": torchvision.__version__},
    }
    if args.ckpt == CKPT:
        META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"selected epoch {best['epoch']} val={meta['val_acc_pct']}% "
          f"train_time={train_time:.0f}s sha256={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
