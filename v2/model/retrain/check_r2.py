#!/usr/bin/env python3
"""V2 Step 2.1b/2.1c: CIFAR-10 r2 acceptance record (r1 vs r2). Label: model.

Run by v2/scripts/regen_results.sh after reference_accuracy.py, requant_check.py
and final_layer.py (it reads their CSVs):

    cd v2/model && PYTHONDONTWRITEBYTECODE=1 ../../.venv/bin/python retrain/check_r2.py

1. Accuracy (legacy code only) of r1 (NET_CONFIGS["cifar10_r1"]) and r2
   (NET_CONFIGS["cifar10"], reference of record since Step 2.1c) on val
   (train[45000:50000], the selection set) and test (10k): FP32 = legacy
   ``cifar10.train.load_checkpoint`` + ``train.evaluate`` on the legacy loaders;
   INT8 = legacy ``calibrate(load_weights(ckpt), train[0:1024])`` +
   ``Int8Cifar10Net.forward`` argmax, and the frozen npz must give the identical
   count. Test counts must equal reference_accuracy.csv.
   -> v2/results/cifar10_r2_accuracy.csv
2. Acceptance rule (DECISIONS D9) -> v2/results/cifar10_r2_summary.csv:
   "accept" only if INT8 test accuracy improves by >= 2.0 pp AND the requant
   check has 0 mismatches at B=32 (cifar10 rows of requant_equivalence.csv), with
   the final-layer check (final_layer_check.csv) recorded alongside.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))
import common  # noqa: E402  (puts legacy python/ on sys.path)
from common import REPO_ROOT, RESULTS_DIR  # noqa: E402

import freeze_cifar10_int8 as fz  # noqa: E402
from net_config import NET_CONFIGS  # noqa: E402

KEYS = {"r1": "cifar10_r1", "r2": "cifar10"}
R2_META = REPO_ROOT / "v2/model/retrain/cifar10_fp32_r2_meta.json"
B_CHECK = 32
MIN_GAIN_PP = 2.0
SPLITS = {"val": (True, range(45000, 50000)), "test": (False, range(10000))}

ACC_CSV = RESULTS_DIR / "cifar10_r2_accuracy.csv"
ACC_FIELDS = ["model", "reference_version", "checkpoint", "split", "precision", "correct",
              "total", "accuracy_pct"]
SUM_CSV = RESULTS_DIR / "cifar10_r2_summary.csv"
SUM_FIELDS = ["quantity", "r1", "r2", "delta", "note"]


def _csv(name) -> list[dict]:
    with open(RESULTS_DIR / name) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- 1. accuracy
def fp32_counts(ckpt_rel: str) -> dict:
    from cifar10 import preprocess, train
    model = train.load_checkpoint(REPO_ROOT / ckpt_rel)
    train_ds, test_ds = preprocess.build_datasets()
    _, val_loader, test_loader = preprocess.fixed_split_loaders(train_ds, test_ds)
    out = {}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        acc, _ = train.evaluate(model, loader, "cpu")
        total = len(loader.dataset)
        correct = int(round(acc * total))
        assert abs(correct / total - acc) < 1e-12
        out[split] = (correct, total)
    return out


def int8_counts(ckpt_rel: str, frozen_npz: Path) -> dict:
    """Legacy-calibrated and frozen-npz INT8 correct counts (must be equal)."""
    from cifar10.int8_model import Int8Cifar10Net
    legacy = Int8Cifar10Net(fz.legacy_params(ckpt_rel=ckpt_rel))
    frozen = Int8Cifar10Net(fz.load_frozen_params(frozen_npz))
    train_ds, test_ds = fz.cifar_datasets()
    out = {}
    for split, (is_train, idx_all) in SPLITS.items():
        ds = train_ds if is_train else test_ds
        idx_all = list(idx_all)
        c_leg = c_frz = 0
        for s in range(0, len(idx_all), 256):
            x, y = fz.stack(ds, idx_all[s:s + 256])
            c_leg += int((legacy.forward(x).argmax(1) == y).sum())
            c_frz += int((frozen.forward(x).argmax(1) == y).sum())
        if c_leg != c_frz:
            raise SystemExit(f"MISMATCH {ckpt_rel} {split}: frozen npz {c_frz} != legacy {c_leg}")
        out[split] = (c_leg, len(idx_all))
    return out


def accuracy() -> dict:
    rows, acc = [], {}
    for name, key in KEYS.items():
        cfg = NET_CONFIGS[key]
        ckpt = cfg["checkpoint"]
        for prec, fn in (("FP32", lambda: fp32_counts(ckpt)),
                         ("INT8", lambda: int8_counts(ckpt, cfg["quant_params"]))):
            t0 = time.perf_counter()
            res = fn()
            dt = round(time.perf_counter() - t0, 3)
            for split, (c, n) in res.items():
                acc[(name, split, prec)] = (c, n)
                r = common.base_meta(net="cifar10", layer="all", source="model",
                                     duration_s=dt, num_inferences=n)
                r.update(model=name, reference_version=cfg["reference_version"], checkpoint=ckpt,
                         split=split, precision=prec, correct=c, total=n,
                         accuracy_pct=f"{100.0 * c / n:.2f}")
                rows.append(r)
                print(f"  {name} {split:4s} {prec} {c}/{n} = {r['accuracy_pct']}%", flush=True)
    common.write_results_csv(ACC_CSV, rows, ACC_FIELDS)
    # Test counts must equal the accuracies of record (both versions).
    rec = {(r["reference_version"], r["precision"]): int(r["correct"])
           for r in _csv("reference_accuracy.csv") if r["net"] == "cifar10"}
    for name, key in KEYS.items():
        ver = NET_CONFIGS[key]["reference_version"]
        for prec in ("FP32", "INT8"):
            assert acc[(name, "test", prec)][0] == rec[(ver, prec)], (name, prec, rec)
    print(f"wrote {ACC_CSV}; test counts equal reference_accuracy.csv")
    return acc


# ---------------------------------------------------------------- 2. checks (from CSVs)
def requant_b32() -> dict:
    rows = [r for r in _csv("requant_equivalence.csv")
            if r["net"] == "cifar10" and int(r["B"]) == B_CHECK]
    assert len(rows) == sum(L["OC"] for L in NET_CONFIGS["cifar10"]["layers"] if not L["final"])
    # Guard against a stale CSV (e.g. r1 rows): the per-channel (m, s) must equal r2's hw_requant.
    with np.load(NET_CONFIGS["cifar10"]["hw_requant"]) as hw:
        assert int(hw["B"]) == B_CHECK
        for r in rows:
            c = int(r["channel"])
            assert (int(r["m"]), int(r["s"])) == (int(hw[f"{r['layer']}_m"][c]),
                                                  int(hw[f"{r['layer']}_s"][c])), r["layer"]
    s = [int(r["s"]) for r in rows]
    return {"mism": sum(int(r["mismatches"]) for r in rows),
            "mism_rne": sum(int(r["mismatches_rne"]) for r in rows),
            "values": sum(int(r["values_checked_exact"]) + int(r["values_checked_saturated"])
                          for r in rows),
            "adjusted": [f"{r['layer']} ch{r['channel']} d={r['delta_from_rne']}" for r in rows
                         if r["delta_from_rne"] not in ("", "0")],
            "smin": min(s), "smax": max(s)}


def final_check() -> dict:
    rows = [r for r in _csv("final_layer_check.csv") if r["net"] == "cifar10"]
    assert len(rows) == 1
    return rows[0]


# ---------------------------------------------------------------- summary
def main() -> int:
    acc = accuracy()
    req = requant_b32()
    fl = final_check()
    meta = json.loads(R2_META.read_text())
    r1_meta = json.loads((REPO_ROOT / "data/checkpoint/cifar10_fp32_meta.json").read_text())
    assert meta["checkpoint"]["sha256"] == common.sha256_file(REPO_ROOT / NET_CONFIGS["cifar10"]["checkpoint"])

    def pct(k):
        c, n = acc[k]
        return 100.0 * c / n

    rows = []

    def add(q, r1, r2, note=""):
        d = f"{r2 - r1:+.2f}" if isinstance(r1, float) and isinstance(r2, float) else ""
        f = (lambda v: f"{v:.2f}" if isinstance(v, float) else v)
        rows.append({**common.base_meta(net="cifar10", layer="all", source="model"),
                     "quantity": q, "r1": f(r1), "r2": f(r2), "delta": d, "note": note})

    for split in ("test", "val"):
        for prec in ("FP32", "INT8"):
            add(f"{prec}_{split}_acc_pct", pct(("r1", split, prec)), pct(("r2", split, prec)))
    add("epochs_run", r1_meta["training"]["epochs"], meta["training"]["epochs_run"])
    add("selected_epoch", r1_meta["training"]["epochs"], meta["selected_epoch"],
        "r1: last epoch (legacy, no selection); r2: argmax val FP32")
    add("train_time_s", "", meta["train_time_s"],
        f"CPU, {meta['torch_threads']} threads (training artifact, r2 meta json)")
    fl_ok = (fl["argmax_match"] == fl["images"] == fl["logits_bitexact"]
             and fl["int32_fits"] == "True")
    add(f"requant_B{B_CHECK}_mismatches", "", req["mism"],
        f"values={req['values']}, mism_rne={req['mism_rne']}, adjusted={req['adjusted']}, "
        f"s={req['smin']}-{req['smax']} (requant_equivalence.csv)")
    add("final_layer_bitexact", "", f"{fl['logits_bitexact']}/{fl['images']}",
        f"argmax {fl['argmax_match']}/{fl['images']}, max|v|={fl['max_abs_v']}, "
        f"int32_fits={fl['int32_fits']} (final_layer_check.csv)")
    gain = pct(("r2", "test", "INT8")) - pct(("r1", "test", "INT8"))
    ok = gain >= MIN_GAIN_PP and req["mism"] == 0
    add("recommendation", "", "accept" if ok else "reject",
        f"rule: INT8 test gain >= {MIN_GAIN_PP} pp ({gain:+.2f}) AND requant B={B_CHECK} "
        f"0 mismatches ({req['mism']}); final layer ok={fl_ok}; adopted in Step 2.1c (D3/D9)")
    common.write_results_csv(SUM_CSV, rows, SUM_FIELDS)
    for r in rows:
        print(f"  {r['quantity']:28s} r1={r['r1']!s:>8s} r2={r['r2']!s:>8s} {r['delta']:>7s} {r['note']}")
    print(f"wrote {SUM_CSV}")
    return 0 if (req["mism"] == 0 and fl_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
