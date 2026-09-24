#!/usr/bin/env python3
"""V2 Step 2.1: freeze the CIFAR-10 INT8 reference parameters (D3).

Run from the repository root:

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/model/freeze_cifar10_int8.py [--out DIR]

Default output: v2/model/frozen/cifar10_int8/{quant_params.npz, manifest.json, SHA256SUMS}.

Step 2.1b: ``--ckpt REL --out DIR`` freezes another FP32 checkpoint of the same
network (e.g. v2/model/retrain/cifar10_fp32_r2.pt -> frozen/cifar10_int8_r2/)
through the identical legacy calibrate path. A non-default checkpoint may not be
written into the default cifar10_int8/ directory.

No reference math is implemented here. The parameters are produced by the legacy
code exactly as python/eval_cifar10_int8.py does:

    w      = cifar10.int8_model.load_weights(data/checkpoint/cifar10_fp32.pt)
    calib  = CIFAR-10 train[0:1024], cifar10.preprocess.make_transform(), float32
    params = cifar10.int8_model.calibrate(w, calib)

and serialized with the key naming of python/export_lenet5_int8.py
(f"{layer}_q_w", "_q_b", "_S_w", "_S_a", "_S_out", "_M", plus "S_input"; the
final ``fc`` layer has no S_out/M). Before writing, the export asserts that
Int8Cifar10Net built from the reloaded npz is bit-identical (every intermediate
and the logits) to the in-memory legacy params on fixed test images.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (puts legacy python/ on sys.path)
from common import FROZEN_DIR, REPO_ROOT, sha256_file  # noqa: E402

import torch  # noqa: E402
import torchvision  # noqa: E402
from torchvision import datasets  # noqa: E402

from cifar10.int8_model import Int8Cifar10Net, calibrate, load_weights  # noqa: E402
from cifar10.preprocess import make_transform  # noqa: E402
from reference import quant  # noqa: E402

SOURCE_TAG = "v1-baseline"
CKPT_REL = "data/checkpoint/cifar10_fp32.pt"
RAW_REL = "data/raw"
CALIB_N = 1024                        # python/eval_cifar10_int8.py CALIB_N
ROUNDTRIP_IDX = list(range(64))       # fixed test images for the npz round-trip check
LAYERS = ("conv1", "conv2", "conv3", "fc")
REQUANT_LAYERS = ("conv1", "conv2", "conv3")
OUT_DIR = FROZEN_DIR / "cifar10_int8"
NPZ_NAME = "quant_params.npz"
MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------- data (legacy path)
def cifar_datasets():
    tr = make_transform()
    root = str(REPO_ROOT / RAW_REL)
    train_ds = datasets.CIFAR10(root=root, train=True, download=False, transform=tr)
    test_ds = datasets.CIFAR10(root=root, train=False, download=False, transform=tr)
    return train_ds, test_ds


def stack(ds, idx) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([ds[i][0].numpy() for i in idx]).astype(np.float32)
    y = np.array([ds[i][1] for i in idx], dtype=np.int64)
    return x, y


def legacy_params(return_weights: bool = False, ckpt_rel: str = CKPT_REL):
    """Exactly the params the legacy INT8 model uses (eval_cifar10_int8.py)."""
    train_ds, _ = cifar_datasets()
    calib, _ = stack(train_ds, range(CALIB_N))
    w = load_weights(REPO_ROOT / ckpt_rel)
    params = calibrate(w, calib)
    return (params, w) if return_weights else params


# ---------------------------------------------------------------- (de)serialization
def params_to_arrays(params: dict) -> dict:
    """Legacy params dict -> flat npz dict (export_lenet5_int8.py key naming)."""
    qp = {"S_input": np.float64(params["S_input"])}
    for name in LAYERS:
        L = params[name]
        qp[f"{name}_q_w"] = L["q_w"]
        qp[f"{name}_q_b"] = L["q_b"]
        qp[f"{name}_S_w"] = L["S_w"].astype(np.float64)
        qp[f"{name}_S_a"] = np.float64(L["S_a"])
        if "S_out" in L:
            qp[f"{name}_S_out"] = np.float64(L["S_out"])
            qp[f"{name}_M"] = L["M"].astype(np.float64)
    return qp


def arrays_to_params(qp) -> dict:
    """Flat npz dict -> params dict accepted by Int8Cifar10Net."""
    p = {"S_input": float(qp["S_input"])}
    for name in LAYERS:
        L = {"S_a": float(qp[f"{name}_S_a"]), "S_w": np.asarray(qp[f"{name}_S_w"]),
             "q_w": np.asarray(qp[f"{name}_q_w"]), "q_b": np.asarray(qp[f"{name}_q_b"])}
        if name in REQUANT_LAYERS:
            L["S_out"] = float(qp[f"{name}_S_out"])
            L["M"] = np.asarray(qp[f"{name}_M"])
        p[name] = L
    return p


def load_frozen_params(path: Path = OUT_DIR / NPZ_NAME) -> dict:
    with np.load(path) as z:
        return arrays_to_params({k: z[k] for k in z.files})


def expected_keys() -> set[str]:
    keys = {"S_input"}
    for name in LAYERS:
        keys |= {f"{name}_{s}" for s in ("q_w", "q_b", "S_w", "S_a")}
        if name in REQUANT_LAYERS:
            keys |= {f"{name}_S_out", f"{name}_M"}
    return keys


# ---------------------------------------------------------------- checks
def check_arrays(qp: dict, params: dict, w: dict) -> None:
    assert set(qp) == expected_keys(), set(qp) ^ expected_keys()
    for name in LAYERS:
        assert qp[f"{name}_q_w"].dtype == np.int8
        assert qp[f"{name}_q_b"].dtype == np.int32
        assert qp[f"{name}_S_w"].dtype == np.float64
        # q_bias fits int32 (ARCH_SPEC numeric contract): the stored int32 equals
        # the float64 RNE value, so the legacy astype(int32) did not wrap.
        L = params[name]
        exact = quant.rne(np.asarray(w[name]["bias"], np.float64) / (L["S_a"] * L["S_w"]))
        assert np.array_equal(exact, qp[f"{name}_q_b"].astype(np.float64)), name
        if name in REQUANT_LAYERS:
            assert qp[f"{name}_M"].dtype == np.float64


def check_roundtrip(params: dict, npz_path: Path) -> int:
    """Bit-identical forward (all intermediates + logits): in-memory vs reloaded."""
    _, test_ds = cifar_datasets()
    x, _ = stack(test_ds, ROUNDTRIP_IDX)
    a = Int8Cifar10Net(params).forward_layers(x)
    b = Int8Cifar10Net(load_frozen_params(npz_path)).forward_layers(x)
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
    return len(ROUNDTRIP_IDX)


# ---------------------------------------------------------------- provenance
def legacy_import_closure() -> list[str]:
    """Legacy python/ source files actually loaded in this process (repo-relative)."""
    legacy = (REPO_ROOT / "python").resolve()
    files = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve().is_relative_to(legacy):
            files.add(Path(f).resolve().relative_to(REPO_ROOT).as_posix())
    return sorted(files)


def source_commit() -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", f"{SOURCE_TAG}^{{commit}}"],
                          capture_output=True, text=True, check=True).stdout.strip()


def unchanged_since_source(paths: list[str]) -> bool:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "diff", "--quiet", SOURCE_TAG, "--", *paths],
                       capture_output=True)
    return r.returncode == 0


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__}


# ---------------------------------------------------------------- export
def export(out_dir: Path = OUT_DIR, ckpt_rel: str = CKPT_REL) -> dict:
    out_dir = Path(out_dir)
    default_ckpt = ckpt_rel == CKPT_REL
    if not default_ckpt and out_dir.resolve() == OUT_DIR.resolve():
        raise SystemExit(f"refusing to overwrite {OUT_DIR} with non-default checkpoint {ckpt_rel}")
    out_dir.mkdir(parents=True, exist_ok=True)
    params, w = legacy_params(return_weights=True, ckpt_rel=ckpt_rel)
    qp = params_to_arrays(params)
    check_arrays(qp, params, w)

    npz = out_dir / NPZ_NAME
    np.savez(npz, **qp)
    n_rt = check_roundtrip(params, npz)

    closure = legacy_import_closure()
    # A retrained checkpoint does not exist at SOURCE_TAG; only the legacy code is
    # compared against it then (the checkpoint is pinned by its sha256 below).
    inputs = ([CKPT_REL] if default_ckpt else []) + closure
    manifest = {
        "artifact": "V2 frozen CIFAR-10 INT8 reference parameters (Step 2.1, D3)" if default_ckpt
                    else "V2 frozen CIFAR-10 INT8 reference parameters, retrained checkpoint "
                         "(Step 2.1b, not yet adopted)",
        "generator": "v2/model/freeze_cifar10_int8.py",
        "source_tag": SOURCE_TAG,
        "source_commit": source_commit(),
        "legacy_inputs_unchanged_since_source_commit": unchanged_since_source(inputs),
        "checkpoint": {"path": ckpt_rel, "sha256": sha256_file(REPO_ROOT / ckpt_rel)},
        "legacy_code": {p: sha256_file(REPO_ROOT / p) for p in closure},
        "calibration": {
            "procedure": "cifar10.int8_model.calibrate(load_weights(checkpoint), calib) "
                         "as in python/eval_cifar10_int8.py",
            "set": "CIFAR-10 train[0:1024] (torchvision CIFAR10, root=data/raw, "
                   "download=False), cifar10.preprocess.make_transform() (ToTensor, "
                   "[0,1], no normalization), stacked as float32",
            "n": CALIB_N,
            "method": "per-tensor activation scale = max|x|/127 (abs-max) of the "
                      "input and of each layer's float ReLU(+pool) output computed "
                      "from the INT8 path; per-output-channel weight scale = "
                      "max|w[c]|/127; q_b = rint(b/(S_a*S_w)) -> int32; M = S_a*S_w/S_out",
            "randomness": "none (fixed contiguous indices; no RNG, seed, shuffle or "
                          "augmentation anywhere in load_weights/calibrate/make_transform)",
        },
        "npz_keys": "export_lenet5_int8.py convention: {layer}_q_w, _q_b, _S_w, _S_a, "
                    "_S_out, _M and S_input; fc (final layer) has no S_out/M",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in qp.items()},
        "roundtrip_check": f"Int8Cifar10Net(npz) == Int8Cifar10Net(in-memory) bit-identical "
                          f"on all forward_layers outputs, test[0:{n_rt}]",
        "outputs": {NPZ_NAME: sha256_file(npz)},
        "versions": versions(),
    }
    man = out_dir / MANIFEST_NAME
    man.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with open(out_dir / "SHA256SUMS", "w") as f:
        for name in sorted([NPZ_NAME, MANIFEST_NAME]):
            f.write(f"{sha256_file(out_dir / name)}  {name}\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--ckpt", default=CKPT_REL, help="FP32 checkpoint, repo-relative")
    ap.add_argument("--check-determinism", action="store_true",
                    help="export twice to temp dirs and compare SHA256")
    args = ap.parse_args()
    if args.check_determinism:
        with tempfile.TemporaryDirectory() as d:
            h = []
            for i in (1, 2):
                o = Path(d) / f"run{i}"
                export(o, args.ckpt)
                h.append({n: sha256_file(o / n) for n in (NPZ_NAME, MANIFEST_NAME)})
            print(f"run1 {h[0]}\nrun2 {h[1]}\nDETERMINISTIC: {h[0] == h[1]}")
            return 0 if h[0] == h[1] else 1
    m = export(args.out, args.ckpt)
    print(f"Wrote {args.out}")
    for n in sorted([NPZ_NAME, MANIFEST_NAME, "SHA256SUMS"]):
        print(f"  {n}  sha256={sha256_file(args.out / n)}")
    print("legacy code closure:", *m["legacy_code"], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
