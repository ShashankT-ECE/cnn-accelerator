#!/usr/bin/env python3
"""V2 Step 2.1: accuracies of record (D3), full 10,000-image test sets. Label: model.

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/model/reference_accuracy.py

Writes v2/results/reference_accuracy.csv (net, precision, correct, total,
accuracy_pct + common.base_meta metadata).

FP32: legacy ``{lenet5,cifar10}.train.load_checkpoint`` + ``train.evaluate`` on
the legacy test loader (``preprocess.build_datasets`` + ``fixed_split_loaders``),
i.e. exactly the test evaluation run at the end of legacy training.

INT8: the legacy runners python/eval_{lenet5,cifar10}_int8.py only print their
counts (main() returns 0), so their loop is reproduced here verbatim with the
legacy functions: calibrate on train[0:1024], build the legacy Int8 model,
argmax of ``model.forward`` over test[0:10000] in the same batches. For CIFAR the
frozen npz (v2/model/frozen/cifar10_int8/quant_params.npz) is evaluated in the
same loop and must give the identical correct count.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import REPO_ROOT, RESULTS_DIR  # noqa: E402

import freeze_cifar10_int8 as fz  # noqa: E402
from torchvision import datasets  # noqa: E402

CSV_PATH = RESULTS_DIR / "reference_accuracy.csv"
FIELDS = ["net", "precision", "correct", "total", "accuracy_pct"]
TEST_N = 10000
CALIB_N = 1024


def _row(net, precision, correct, total, t0) -> dict:
    r = common.base_meta(net=net, layer="all", source="model",
                         duration_s=round(time.perf_counter() - t0, 3), num_inferences=total)
    r.update(net=net, precision=precision, correct=int(correct), total=int(total),
             accuracy_pct=f"{100.0 * correct / total:.2f}")
    print(f"  {net:8s} {precision:5s} {correct}/{total} = {r['accuracy_pct']}%  "
          f"({r['duration_s']} s)", flush=True)
    return r


def _fp32(train_mod, preprocess_mod, ckpt_rel: str) -> tuple[int, int]:
    model = train_mod.load_checkpoint(REPO_ROOT / ckpt_rel)
    train_ds, test_ds = preprocess_mod.build_datasets()
    _, _, test_loader = preprocess_mod.fixed_split_loaders(train_ds, test_ds)
    acc, _ = train_mod.evaluate(model, test_loader, "cpu")
    total = len(test_loader.dataset)
    correct = int(round(acc * total))
    assert abs(correct / total - acc) < 1e-12
    return correct, total


def lenet_fp32() -> dict:
    from lenet5 import preprocess, train
    t0 = time.perf_counter()
    return _row("lenet5", "FP32", *_fp32(train, preprocess, "data/checkpoint/lenet5_fp32.pt"), t0)


def cifar_fp32() -> dict:
    from cifar10 import preprocess, train
    t0 = time.perf_counter()
    return _row("cifar10", "FP32", *_fp32(train, preprocess, "data/checkpoint/cifar10_fp32.pt"), t0)


def lenet_int8() -> dict:
    # Mirrors python/eval_lenet5_int8.py (BATCH = 512).
    from lenet5.int8_model import Int8LeNet5, calibrate, load_weights
    from lenet5.preprocess import make_transform
    t0 = time.perf_counter()
    tr = make_transform()
    train_ds = datasets.MNIST(root=str(REPO_ROOT / "data/raw"), train=True,
                              download=False, transform=tr)
    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data/raw"), train=False,
                             download=False, transform=tr)
    calib = np.stack([train_ds[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)
    model = Int8LeNet5(calibrate(load_weights(REPO_ROOT / "data/checkpoint/lenet5_fp32.pt"), calib))
    correct = 0
    for s in range(0, TEST_N, 512):
        x = np.stack([test_ds[i][0].numpy() for i in range(s, min(s + 512, TEST_N))])
        x = x.astype(np.float32)
        y = np.array([test_ds[i][1] for i in range(s, min(s + 512, TEST_N))])
        correct += int((model.forward(x).argmax(1) == y).sum())
    return _row("lenet5", "INT8", correct, TEST_N, t0)


def cifar_int8_counts() -> tuple[int, int, float]:
    """(legacy-calibrated correct, frozen-npz correct, seconds). Mirrors eval_cifar10_int8.py."""
    from cifar10.int8_model import Int8Cifar10Net
    t0 = time.perf_counter()
    legacy = Int8Cifar10Net(fz.legacy_params())
    frozen = Int8Cifar10Net(fz.load_frozen_params())
    _, test_ds = fz.cifar_datasets()
    c_leg = c_frz = 0
    for s in range(0, TEST_N, 256):
        idx = list(range(s, min(s + 256, TEST_N)))
        x = np.stack([test_ds[i][0].numpy() for i in idx]).astype(np.float32)
        y = np.array([test_ds[i][1] for i in idx])
        c_leg += int((legacy.forward(x).argmax(1) == y).sum())
        c_frz += int((frozen.forward(x).argmax(1) == y).sum())
    return c_leg, c_frz, t0


def main() -> int:
    rows = [lenet_fp32(), lenet_int8(), cifar_fp32()]
    c_leg, c_frz, t0 = cifar_int8_counts()
    rows.append(_row("cifar10", "INT8", c_leg, TEST_N, t0))
    print(f"  cifar10 INT8 frozen npz: {c_frz}/{TEST_N} (legacy calibrate: {c_leg})")
    if c_frz != c_leg:
        print("MISMATCH: frozen npz != legacy calibrate", file=sys.stderr)
        return 1
    common.write_results_csv(CSV_PATH, rows, FIELDS)
    print(f"Wrote {CSV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
