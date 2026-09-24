#!/usr/bin/env python3
"""V2 Step 2.1b: evaluate the retrained CIFAR-10 r2 reference. Label: model.

Run from the repository root (after train_cifar10_r2.py and
``freeze_cifar10_int8.py --ckpt v2/model/retrain/cifar10_fp32_r2.pt --out v2/model/frozen/cifar10_int8_r2``):

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/model/retrain/check_r2.py

1. Accuracy (legacy code only): FP32 = legacy ``cifar10.train.load_checkpoint`` +
   ``train.evaluate`` on the legacy val/test loaders; INT8 = legacy
   ``calibrate(load_weights(ckpt), train[0:1024])`` + ``Int8Cifar10Net.forward``
   argmax, and the frozen r2 npz must give the identical count. Both the current
   reference (r1, data/checkpoint/cifar10_fp32.pt) and r2 are evaluated on val
   (train[45000:50000]) and test (10k); r1 test must equal reference_accuracy.csv.
   -> v2/results/cifar10_r2_accuracy.csv
2. Step 2.1 requant equivalence (requant_check.run, feasible-m search) on the r2
   npz at B=32 -> v2/results/requant_equivalence_r2.csv; if 0 mismatches the
   per-channel (m, s) are written to frozen/cifar10_int8_r2/hw_requant.npz.
3. Step 2.1 final-layer test (final_layer.check_net on the r2 checkpoint)
   -> v2/results/final_layer_check_r2.csv
4. Acceptance rule -> v2/results/cifar10_r2_summary.csv: recommend "accept" only
   if INT8 test accuracy improves by >= 2.0 pp AND requant B=32 has 0 mismatches.
   Nothing is switched: the reference of record stays cifar10_int8/.
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
from common import FROZEN_DIR, REPO_ROOT, RESULTS_DIR  # noqa: E402

import final_layer  # noqa: E402
import freeze_cifar10_int8 as fz  # noqa: E402
import requant_check  # noqa: E402

R1_CKPT = fz.CKPT_REL
R2_CKPT = "v2/model/retrain/cifar10_fp32_r2.pt"
R2_META = REPO_ROOT / "v2/model/retrain/cifar10_fp32_r2_meta.json"
R2_DIR = FROZEN_DIR / "cifar10_int8_r2"
R2_NPZ = R2_DIR / fz.NPZ_NAME
R2_HW_NPZ = R2_DIR / "hw_requant.npz"
B_CHECK = 32
MIN_GAIN_PP = 2.0
SPLITS = {"val": (True, range(45000, 50000)), "test": (False, range(10000))}

ACC_CSV = RESULTS_DIR / "cifar10_r2_accuracy.csv"
ACC_FIELDS = ["model", "checkpoint", "split", "precision", "correct", "total", "accuracy_pct"]
REQ_CSV = RESULTS_DIR / "requant_equivalence_r2.csv"
FL_CSV = RESULTS_DIR / "final_layer_check_r2.csv"
SUM_CSV = RESULTS_DIR / "cifar10_r2_summary.csv"
SUM_FIELDS = ["quantity", "r1", "r2", "delta", "note"]


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
    for name, ckpt, npz in (("r1", R1_CKPT, fz.OUT_DIR / fz.NPZ_NAME), ("r2", R2_CKPT, R2_NPZ)):
        for prec, fn in (("FP32", lambda: fp32_counts(ckpt)), ("INT8", lambda: int8_counts(ckpt, npz))):
            t0 = time.perf_counter()
            res = fn()
            dt = round(time.perf_counter() - t0, 3)
            for split, (c, n) in res.items():
                acc[(name, split, prec)] = (c, n)
                r = common.base_meta(net="cifar10", layer="all", source="model",
                                     duration_s=dt, num_inferences=n)
                r.update(model=name, checkpoint=ckpt, split=split, precision=prec, correct=c,
                         total=n, accuracy_pct=f"{100.0 * c / n:.2f}")
                rows.append(r)
                print(f"  {name} {split:4s} {prec} {c}/{n} = {r['accuracy_pct']}%", flush=True)
    common.write_results_csv(ACC_CSV, rows, ACC_FIELDS)
    # r1 test must reproduce the accuracies of record.
    with open(RESULTS_DIR / "reference_accuracy.csv") as f:
        rec = {row["precision"]: int(row["correct"]) for row in csv.DictReader(f)
               if row["net"] == "cifar10"}
    for prec in ("FP32", "INT8"):
        assert acc[("r1", "test", prec)][0] == rec[prec], (prec, acc[("r1", "test", prec)], rec)
    print(f"wrote {ACC_CSV}; r1 test reproduces reference_accuracy.csv")
    return acc


# ---------------------------------------------------------------- 2. requant
def requant() -> dict:
    res = requant_check.run(["cifar10"], {"cifar10": R2_NPZ}, bits=(B_CHECK,),
                            out_csv=REQ_CSV, save=False)
    agg = res["summary"][("cifar10", B_CHECK)]
    if agg["mism"] == 0:
        sel = {}
        for L in requant_check.load_layers("cifar10", R2_NPZ):
            by = {r["channel"]: r for r in res["rows"] if r["layer"] == L["layer"]}
            ms = [(by[c]["m"], by[c]["s"]) for c in range(L["M"].size)]
            assert all(by[c]["mismatches"] == 0 for c in by)
            sel[f"{L['layer']}_m"] = np.array([m for m, _ in ms], dtype=np.uint64)
            sel[f"{L['layer']}_s"] = np.array([s for _, s in ms], dtype=np.uint8)
            assert all(int(x) == m for x, (m, _) in zip(sel[f"{L['layer']}_m"], ms))
        sel["B"] = np.array(B_CHECK, dtype=np.int64)
        np.savez(R2_HW_NPZ, **sel)
        print(f"wrote {R2_HW_NPZ.relative_to(REPO_ROOT)} sha256={common.sha256_file(R2_HW_NPZ)}")
    return agg


# ---------------------------------------------------------------- 3. final layer
def final() -> dict:
    t0 = time.time()
    # Guard: the final-layer check must run on r2 (params equal the frozen r2 npz).
    P = final_layer.calibrated_params("cifar10", R2_CKPT)["fc"]
    with np.load(R2_NPZ) as z:
        assert np.array_equal(P["q_w"], z["fc_q_w"]) and np.array_equal(P["q_b"], z["fc_q_b"])
    r = final_layer.check_net("cifar10", ckpt=R2_CKPT)
    meta = common.base_meta(net="cifar10", layer=r["layer"], source="model",
                            duration_s=round(time.time() - t0, 2), num_inferences=r["images"])
    common.write_results_csv(FL_CSV, [{**meta, **{k: r[k] for k in final_layer.CSV_FIELDS}}],
                             final_layer.CSV_FIELDS)
    print(r, f"\nwrote {FL_CSV}")
    return r


# ---------------------------------------------------------------- 4. summary
def main() -> int:
    acc = accuracy()
    req = requant()
    fl = final()
    meta = json.loads(R2_META.read_text())
    r1_meta = json.loads((REPO_ROOT / "data/checkpoint/cifar10_fp32_meta.json").read_text())

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
    add("train_time_s", "", meta["train_time_s"], f"CPU, {meta['torch_threads']} threads")
    fl_ok = fl["argmax_match"] == fl["images"] == fl["logits_bitexact"] and fl["int32_fits"]
    add(f"requant_B{B_CHECK}_mismatches", "", req["mism"],
        f"values={req['exact'] + req['sat']}, mism_rne={req['mism_rne']}, "
        f"adjusted_ch={req['adjusted']}, max|d|={req['max_abs_delta']}, s={req['smin']}-{req['smax']}")
    add("final_layer_bitexact", "", f"{fl['logits_bitexact']}/{fl['images']}",
        f"argmax {fl['argmax_match']}/{fl['images']}, max|v|={fl['max_abs_v']}, "
        f"int32_fits={fl['int32_fits']}")
    gain = pct(("r2", "test", "INT8")) - pct(("r1", "test", "INT8"))
    ok = gain >= MIN_GAIN_PP and req["mism"] == 0
    add("recommendation", "", "accept" if ok else "reject",
        f"rule: INT8 test gain >= {MIN_GAIN_PP} pp ({gain:+.2f}) AND requant B={B_CHECK} "
        f"0 mismatches ({req['mism']}); final layer ok={fl_ok}; not adopted")
    common.write_results_csv(SUM_CSV, rows, SUM_FIELDS)
    for r in rows:
        print(f"  {r['quantity']:28s} r1={r['r1']!s:>8s} r2={r['r2']!s:>8s} {r['delta']:>7s} {r['note']}")
    print(f"wrote {SUM_CSV}")
    return 0 if (req["mism"] == 0 and fl_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
