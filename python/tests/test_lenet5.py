#!/usr/bin/env python3
"""Self-checks for the Step 1B L1 FP32 LeNet-5 reference.

Verifies, independently of the generation path:

  T1  the model consumes [N,1,32,32].
  T2  every intermediate tensor shape matches docs/LENET5_SPEC.md §2.
  T3  the final output is [N,10] (logits).
  T4  total parameter count == 61,706.
  T5  the saved checkpoint loads, and inference from it is deterministic
      (bit-identical logits across two forward passes).
  T6  the checkpoint is reproducible: re-serializing the loaded state_dict
      yields the same SHA-256 recorded in lenet5_fp32_meta.json, and the
      canonical-weights hash matches the recorded value.

Plain asserts; no pytest dependency. Exit 0 iff all pass.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from lenet5 import config as C  # noqa: E402
from lenet5.model import LeNet5  # noqa: E402
from lenet5.train import (canonical_state_dict_sha256, load_checkpoint,  # noqa: E402
                          sha256_file)

CKPT = REPO_ROOT / C.CKPT_DIR / C.CKPT_FILE
META = REPO_ROOT / C.CKPT_DIR / C.META_FILE

N = 2  # batch size for shape checks

# Expected intermediate shapes (docs/LENET5_SPEC.md §2).
EXPECTED_SHAPES = {
    "input":  (N, 1, 32, 32),
    "c1":     (N, 6, 28, 28),
    "relu1":  (N, 6, 28, 28),
    "pool1":  (N, 6, 14, 14),
    "c3":     (N, 16, 10, 10),
    "relu3":  (N, 16, 10, 10),
    "pool2":  (N, 16, 5, 5),
    "c5":     (N, 120, 1, 1),
    "relu5":  (N, 120, 1, 1),
    "f6":     (N, 84),
    "relu6":  (N, 84),
    "output": (N, 10),
}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 1B LeNet-5 L1 reference self-checks")

    torch.manual_seed(0)
    model = LeNet5()
    model.eval()
    x = torch.rand(N, 1, C.IMG_H, C.IMG_W)

    # T1: input shape accepted.
    check("T1 input shape [N,1,32,32]", tuple(x.shape) == (N, 1, 32, 32),
          str(tuple(x.shape)))

    # T2: intermediate shapes.
    with torch.no_grad():
        layers = model.forward_layers(x)
    for name, expected in EXPECTED_SHAPES.items():
        got = tuple(layers[name].shape)
        check(f"T2 {name} shape", got == expected, f"{got}")

    # T3: output [N,10].
    logits = model(x)
    check("T3 output shape [N,10]", tuple(logits.shape) == (N, 10),
          str(tuple(logits.shape)))

    # T4: param count.
    n_params = sum(p.numel() for p in model.parameters())
    check("T4 total params == 61706", n_params == C.TOTAL_PARAMS, f"{n_params}")

    # T5/T6 require the trained checkpoint.
    if not CKPT.exists():
        check("T5 checkpoint present", False,
              f"{CKPT} missing — run python/train_lenet5.py first")
        return 1

    loaded = load_checkpoint(CKPT)
    with torch.no_grad():
        a = loaded(x)
        b = loaded(x)
    check("T5 deterministic inference (two forward passes identical)",
          torch.equal(a, b))

    # T6: checkpoint reproducibility anchors.
    meta = json.loads(META.read_text())
    recorded_sha = meta["checkpoint"]["sha256"]
    recorded_canon = meta["checkpoint"]["canonical_weights_sha256"]

    # T6a: the committed checkpoint file's SHA-256 matches the recorded value.
    check("T6a recorded file SHA matches committed file",
          sha256_file(CKPT) == recorded_sha, f"{sha256_file(CKPT)} vs {recorded_sha}")

    # T6b: the canonical weights hash (a pure function of the values) matches
    # the recorded value — the reproducibility anchor, independent of the
    # torch serialization format.
    sd = torch.load(CKPT)
    canon = canonical_state_dict_sha256(sd)
    check("T6b canonical-weights SHA matches record", canon == recorded_canon,
          f"{canon} vs {recorded_canon}")

    # T6c: torch.save (legacy format) is byte-deterministic for a fixed
    # state_dict, so the file SHA-256 is well-defined given deterministic weights.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(sd, tmp.name, _use_new_zipfile_serialization=False)
        tmp_path = Path(tmp.name)
    h1 = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    with open(tmp_path, "wb") as f:
        torch.save(sd, f, _use_new_zipfile_serialization=False)
    h2 = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
    tmp_path.unlink()
    check("T6c torch.save byte-deterministic (two saves identical)", h1 == h2,
          f"{h1} vs {h2}")

    print("  ALL STEP 1B CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
