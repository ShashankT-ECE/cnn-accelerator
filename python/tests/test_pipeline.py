#!/usr/bin/env python3
"""End-to-end self-checks for the Phase 1 pipeline.

Verifies, independently of the generation path:

  T1  the independent golden model: pure-Python loop == vectorised einsum
      (this is the guard against a "golden model written in the datapath's own
      language" — Decision 9 / RESEARCH_READINESS_PLAN §10).
  T2  the exported weights.hex / input_img.hex parse to the expected tensors.
  T3  the exported golden_canonical.txt equals the loop golden (bit-exact).
  T4  the golden fits int32 and matches the shape 8x28x28.
  T5  the manifest is internally consistent (shapes, counts, scales).

Plain asserts; no pytest dependency. Exit 0 iff all pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from reference import constants as C, int8_ref, onnx_ref, quant  # noqa: E402

VECTORS = REPO_ROOT / "data" / "vectors"


def parse_hex(path: Path) -> np.ndarray:
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        u = int(line, 16)
        vals.append(u - 256 if u >= 128 else u)  # two's-complement -> signed
    return np.array(vals, dtype=np.int64)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Phase 1 pipeline self-checks")

    # T1: golden loop vs einsum (random + deterministic).
    rng = np.random.default_rng(0)
    for trial in range(5):
        qa = rng.integers(-128, 128, size=(28, 28), dtype=np.int8)
        qw = rng.integers(-128, 128, size=(8, 5, 5), dtype=np.int8)
        a = int8_ref.conv1_golden_loop(qa, qw)
        b = int8_ref.conv1_golden_einsum(qa, qw)
        if not np.array_equal(a, b):
            check("T1 golden loop==einsum", False, f"trial {trial}")
            return 1
    check("T1 golden loop==einsum", True, "5 random trials + primary vector")

    # Load exported artifacts.
    weights = parse_hex(VECTORS / "weights.hex")
    img = parse_hex(VECTORS / "input_img.hex")
    golden = np.loadtxt(VECTORS / "golden_canonical.txt", dtype=np.int64)
    manifest = json.loads((VECTORS / "manifest.json").read_text())

    # T2: hex parse shapes.
    check("T2 weights.hex shape", weights.shape == (C.CONV1_WEIGHTS,),
          f"{weights.shape}")
    check("T2 input_img.hex shape", img.shape == (C.IMG_H * C.IMG_W,), f"{img.shape}")

    # T3: golden == loop golden for the primary image.
    q_w = weights.reshape(C.CONV1_OC, C.CONV1_K, C.CONV1_K).astype(np.int8)
    q_a = img.reshape(C.IMG_H, C.IMG_W).astype(np.int8)
    golden_loop = int8_ref.conv1_golden_loop(q_a, q_w).ravel()
    check("T3 golden_canonical == loop golden", np.array_equal(golden, golden_loop),
          f"{golden.shape} vs {golden_loop.shape}")

    # T4: golden fits int32 and has the right shape.
    check("T4 golden shape", golden.shape == (C.CONV1_OUTPUTS,), f"{golden.shape}")
    check("T4 golden fits int32",
          golden.min() >= -2**31 and golden.max() <= 2**31 - 1,
          f"range [{golden.min()}, {golden.max()}]")

    # T5: manifest consistency.
    al = manifest["accelerated_layer"]
    check("T5 manifest out_channels", al["out_channels"] == C.CONV1_OC)
    check("T5 manifest outputs", al["outputs"] == C.CONV1_OUTPUTS)
    check("T5 manifest maccs", al["maccs"] == C.CONV1_MACS)
    q = manifest["quantization"]
    check("T5 S_a == 1/127", abs(q["S_a"] - (1.0 / 127.0)) < 1e-18, repr(q["S_a"]))
    check("T5 zero-points are 0", q["zero_point_activation"] == 0 and q["zero_point_weight"] == 0)

    # Cross-check quant_params.npz matches manifest.
    p = np.load(VECTORS / "quant_params.npz")
    check("T5 npz q_w == manifest", np.array_equal(p["q_w"].ravel(), weights))

    print("  ALL PHASE 1 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
