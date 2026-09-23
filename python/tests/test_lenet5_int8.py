#!/usr/bin/env python3
"""Self-checks for the Step 2.2 L2 INT8 hardware-reference model.

Verifies, independently of the generation path:

  T1  quantization primitives (RNE, scales, per-channel, bias).
  T2  requantization (canonical RNE) and fixed-point realization (≤1 LSB apart).
  T3  integer conv (im2col+einsum) == a manual nested-loop reference.
  T4  integer max-pool.
  T5  the full INT8 forward: every layer shape, int8 activation dtypes, int32
      accumulation, dequantized logits.
  T6  INT8 (L2) logits agree with the FP32 (L1) logits (argmax on a sample).

Plain asserts; no pytest dependency. Exit 0 iff all pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import torch  # noqa: E402
from torchvision import datasets  # noqa: E402

from reference import quant  # noqa: E402
from lenet5.int8_quant import (requantize, requantize_fixed, rne,  # noqa: E402
                               per_channel_scales, tensor_scale_abs_max,
                               quantize_bias_int32, quantize_per_channel,
                               quantize_tensor)
from lenet5.int8_model import int_conv2d, int_maxpool2d  # noqa: E402
from lenet5.preprocess import make_transform  # noqa: E402
from lenet5.train import load_checkpoint  # noqa: E402
from lenet5.int8_model import Int8LeNet5, calibrate, load_weights  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def conv_loop(q_a: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    """Manual integer valid conv (definition) — reference for int_conv2d."""
    N, IC, H, W = q_a.shape
    OC, ICw, K, K = q_w.shape
    OH, OW = H - K + 1, W - K + 1
    out = np.zeros((N, OC, OH, OW), dtype=np.int64)
    for n in range(N):
        for o in range(OC):
            for i in range(OH):
                for j in range(OW):
                    s = 0
                    for ic in range(IC):
                        for ky in range(K):
                            for kx in range(K):
                                s += int(q_a[n, ic, i + ky, j + kx]) * int(q_w[o, ic, ky, kx])
                    out[n, o, i, j] = s
    return out


def main() -> int:
    print("Step 2.2 LeNet-5 L2 INT8 reference self-checks")

    # ---- T1: quantization primitives ----
    check("T1 rne round-half-to-even", (rne(np.array([0.5, 1.5, 2.5, 3.5, -0.5, -1.5]))
          == np.array([0, 2, 2, 4, 0, -2])).all())
    x = np.array([0.0, 0.5, 1.0, -0.25, -0.5, 0.125])
    s = tensor_scale_abs_max(x)
    check("T1 tensor scale == max|x|/127", abs(s - 1.0 / 127.0) < 1e-15, repr(s))
    w = np.array([[[0.1]], [[-0.2]], [[0.05]]])  # (3,1,1) per-channel
    sw = per_channel_scales(w)
    check("T1 per-channel scales", (np.abs(sw - np.array([0.1, 0.2, 0.05]) / 127.0) < 1e-12).all())
    qw, _ = quantize_per_channel(w)
    check("T1 quantize_per_channel int8", qw.dtype == np.int8 and qw.shape == w.shape)
    qb = quantize_bias_int32(np.array([0.1, 0.2, 0.05]), 1.0 / 127.0, sw)
    check("T1 bias int32", qb.dtype == np.int32 and qb.shape == (3,))
    q = quantize_tensor(np.array([0.0, 1.0]), 1.0 / 127.0)
    check("T1 quantize_tensor [0,127]", q.tolist() == [0, 127], str(q.tolist()))

    # ---- T2: requantization ----
    y = np.array([100, -100, 0, 1234, -5678])
    M = np.array([0.5, 1.0, 1.5, 0.25, 2.0])
    qr = requantize(y, M)
    expect = np.clip(np.rint(y.astype(np.float64) * M), -128, 127).astype(np.int8)
    check("T2 canonical requantize == rne(y·M)", (qr == expect).all(), str(qr.tolist()))
    rng = np.random.default_rng(0)
    yr = rng.integers(-100000, 100000, size=1000).astype(np.int64)
    Mr = 10 ** rng.uniform(-1, 0.3, size=1000)  # M in [0.1, 2)
    qc = requantize(yr, Mr)
    qf = requantize_fixed(yr, Mr)
    d = int(np.abs(qc.astype(np.int32) - qf.astype(np.int32)).max())
    check("T2 fixed-point == canonical within 1 LSB", d <= 1, f"max|diff|={d}")

    # ---- T3: integer conv vs manual loop ----
    rng = np.random.default_rng(1)
    for (N, IC, H, W, K, OC) in [(2, 1, 8, 8, 3, 4), (2, 3, 6, 6, 3, 5)]:
        qa = rng.integers(-128, 128, size=(N, IC, H, W)).astype(np.int8)
        qw = rng.integers(-128, 128, size=(OC, IC, K, K)).astype(np.int8)
        a = int_conv2d(qa, qw)
        b = conv_loop(qa, qw)
        check(f"T3 int_conv2d == loop (IC={IC})", np.array_equal(a, b), f"shape {a.shape}")

    # ---- T4: integer max-pool ----
    x = np.array([[[[1, 5, 2, 6], [3, 4, 8, 7], [9, 1, 2, 3], [4, 6, 5, 8]]]], dtype=np.int8)
    p = int_maxpool2d(x)
    check("T4 int_maxpool2d", p.tolist() == [[[[5, 8], [9, 8]]]], str(p.tolist()))

    # ---- T5/T6: full model (calibrate on a small subset; compare to L1) ----
    ckpt = REPO_ROOT / "data/checkpoint/lenet5_fp32.pt"
    w = load_weights(ckpt)
    tr = make_transform()
    train_ds = datasets.MNIST(root=str(REPO_ROOT / "data/raw"), train=True,
                              download=False, transform=tr)
    calib = np.stack([train_ds[i][0].numpy() for i in range(256)]).astype(np.float32)
    params = calibrate(w, calib)
    model = Int8LeNet5(params)

    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data/raw"), train=False,
                             download=False, transform=tr)
    N = 8
    xf = np.stack([test_ds[i][0].numpy() for i in range(N)]).astype(np.float32)
    ys = np.array([test_ds[i][1] for i in range(N)])
    out = model.forward_layers(xf)

    shapes = {
        "q_input": (N, 1, 32, 32), "c1": (N, 6, 28, 28), "pool1": (N, 6, 14, 14),
        "c3": (N, 16, 10, 10), "pool2": (N, 16, 5, 5), "c5": (N, 120, 1, 1),
        "f6": (N, 84), "logits": (N, 10),
    }
    for k, sh in shapes.items():
        check(f"T5 {k} shape", out[k].shape == sh, str(out[k].shape))
    for k in ("q_input", "c1", "pool1", "c3", "pool2", "c5", "f6"):
        check(f"T5 {k} dtype int8", out[k].dtype == np.int8, str(out[k].dtype))
    for k in ("c1_acc", "c3_acc", "c5_acc", "f6_acc", "out_acc"):
        acc = out[k]
        check(f"T5 {k} fits int32", int(acc.min()) >= -2**31 and int(acc.max()) <= 2**31 - 1,
              f"range [{int(acc.min())}, {int(acc.max())}]")
    check("T5 logits float32", out["logits"].dtype == np.float32)

    # Cross-check L2 (INT8) vs L1 (FP32) argmax agreement on a 256-image sample.
    l1 = load_checkpoint(ckpt)
    xs = torch.stack([test_ds[i][0] for i in range(256)])
    with torch.no_grad():
        l1_logits = l1(xs).numpy()
    xf256 = np.stack([test_ds[i][0].numpy() for i in range(256)]).astype(np.float32)
    l2_logits = model.forward(xf256)
    agree = int((l1_logits.argmax(1) == l2_logits.argmax(1)).sum())
    check("T6 INT8 vs FP32 argmax agreement >= 95%", agree >= int(256 * 0.95),
          f"{agree}/256")

    print("  ALL STEP 2.2 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
