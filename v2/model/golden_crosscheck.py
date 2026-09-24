"""Golden-vs-legacy INT8 cross-check (V2 step 2.2 part A) as a results script.

Compares gos_golden (integer-only, hardware requant) with the legacy INT8
reference per image and per layer on the full test sets and writes
v2/results/golden_crosscheck.csv (source=model). The legacy model is built from
the NET_CONFIGS quant_params npz (not via calibrate), so a retrained parameter
set works unchanged. Legacy intermediates used:

  LeNet-5  python/lenet5/int8_model.py : q_input :146, pool1 :154, pool2 :162,
           c5 :168, f6 :175, out_acc :179 (v = out_acc + q_b), logits :181
  CIFAR-10 python/cifar10/int8_model.py: q_input :85, pool1 :93, pool2 :101,
           c3 :108, fc_acc :113 (v = fc_acc + q_b), logits :115

Usage: python golden_crosscheck.py [--n 10000]
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache

import numpy as np

import common  # noqa: F401  (legacy python/ on sys.path)
from common import RESULTS_DIR, base_meta, write_results_csv
import gos_golden as gg
from net_config import NET_CONFIGS

NETS = ("lenet5", "cifar10")
TEST_N = 10000
BATCH = 500
ACCURACY_OF_RECORD = {"lenet5": 9879, "cifar10": 6576}   # DECISIONS D3 outcome

# golden layer name -> legacy forward_layers key (int8 outputs; final = acc key)
LEGACY_KEYS = {
    "lenet5": {"conv1": "pool1", "conv3": "pool2", "conv5": "c5", "fc1": "f6",
               "fc2": "out_acc"},
    "cifar10": {"conv1": "pool1", "conv2": "pool2", "conv3": "c3", "fc": "fc_acc"},
}
# keys legacy calibrate() produces per layer (the model reads a subset of them)
NONFINAL_KEYS = {"S_a", "S_w", "q_w", "q_b", "S_out", "M"}
FINAL_KEYS = {"S_a", "S_w", "q_w", "q_b"}


# --------------------------------------------------------------------------- #
# Legacy model from the NET_CONFIGS npz
# --------------------------------------------------------------------------- #
def legacy_params(net: str) -> dict:
    z = np.load(NET_CONFIGS[net]["quant_params"])
    p = {"S_input": float(z["S_input"])}
    names = [L["name"] for L in NET_CONFIGS[net]["layers"]]
    for L in NET_CONFIGS[net]["layers"]:
        n = L["name"]
        keys = {k[len(n) + 1:] for k in z.files if k.startswith(n + "_")}
        want = FINAL_KEYS if L["final"] else NONFINAL_KEYS
        assert keys == want, (net, n, keys, want)
        p[n] = {k: (float(z[f"{n}_{k}"]) if z[f"{n}_{k}"].ndim == 0 else z[f"{n}_{k}"])
                for k in keys}
    assert set(z.files) == {"S_input"} | {f"{n}_{k}" for n in names for k in p[n]}
    return p


def legacy_model(net: str):
    if net == "lenet5":
        from lenet5.int8_model import Int8LeNet5 as Model, _LAYERS
    else:
        from cifar10.int8_model import Int8Cifar10Net as Model, _LAYERS
    names = tuple(L["name"] for L in NET_CONFIGS[net]["layers"])
    assert names == tuple(_LAYERS), (names, _LAYERS)
    return Model(legacy_params(net))


@lru_cache(maxsize=None)
def _test_set(net: str, n: int = TEST_N):
    return gg.load_test_set(net, n)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare(net: str, n: int, batch: int = BATCH) -> dict:
    G = gg.load_net(net)
    model = legacy_model(net)
    keys = LEGACY_KEYS[net]
    x_all, y_all = _test_set(net, TEST_N) if n > 64 else _test_set(net, n)
    x_all, y_all = x_all[:n], y_all[:n]
    names = [P.name for P in G.layers]
    exact = {k: 0 for k in ["input"] + names}
    pred_eq = correct = legacy_correct = 0
    t_gold = t_leg = 0.0
    for s0 in range(0, n, batch):
        x, y = x_all[s0:s0 + batch], y_all[s0:s0 + batch]
        nb = x.shape[0]
        t = time.time()
        ref = model.forward_layers(x)
        t_leg += time.time() - t
        t = time.time()
        q = gg.quantize_input(G, x)
        out = gg.run_net(G, q)
        t_gold += time.time() - t

        assert q.dtype == np.int8 and ref["q_input"].dtype == np.int8
        exact["input"] += int(np.all((q == ref["q_input"]).reshape(nb, -1), 1).sum())
        for P in G.layers:
            g = out[P.name]
            r = ref[keys[P.name]]
            if P.cfg["final"]:
                assert g.dtype == np.int32
                r = r.astype(np.int64) + P.q_b.astype(np.int64).reshape(1, -1)
                g = g.reshape(nb, -1).astype(np.int64)
                assert np.array_equal(out["v"].astype(np.int64), g)
            else:
                assert g.dtype == np.int8 and r.dtype == np.int8, (P.name, g.dtype, r.dtype)
                g = g.reshape(nb, -1)
                assert g.shape == r.reshape(nb, -1).shape, (P.name, g.shape, r.shape)
            exact[P.name] += int(np.all(g == r.reshape(nb, -1), 1).sum())
        ref_pred = ref["logits"].argmax(1)
        assert np.array_equal(out["logits"].view(np.uint32), ref["logits"].view(np.uint32))
        pred_eq += int((out["pred"] == ref_pred).sum())
        correct += int((out["pred"] == y).sum())
        legacy_correct += int((ref_pred == y).sum())
    return {"net": net, "images": n, "exact": exact, "pred_eq": pred_eq,
            "correct": correct, "legacy_correct": legacy_correct,
            "t_golden_s": round(t_gold, 2), "t_legacy_s": round(t_leg, 2)}



CSV_PATH = RESULTS_DIR / "golden_crosscheck.csv"
CSV_FIELDS = ["images", "bit_identical", "predictions_identical", "golden_correct",
              "legacy_correct", "compared"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=TEST_N)
    a = ap.parse_args(argv)
    rows = []
    ok = True
    for net in NETS:
        t0 = time.time()
        r = compare(net, a.n)
        dt = round(time.time() - t0, 1)
        for layer, cnt in r["exact"].items():
            final = layer == NET_CONFIGS[net]["layers"][-1]["name"]
            row = base_meta(net=net, layer=layer, source="model", duration_s=dt,
                            num_inferences=r["images"])
            row.update(images=r["images"], bit_identical=cnt,
                       predictions_identical=r["pred_eq"], golden_correct=r["correct"],
                       legacy_correct=r["legacy_correct"],
                       compared="quantized input" if layer == "input" else
                       ("raw int32 v = acc + q_bias" if final else "int8 layer output"))
            rows.append(row)
            ok &= cnt == r["images"]
        ok &= r["pred_eq"] == r["images"] and r["correct"] == r["legacy_correct"]
        print(f"{net}: {r['images']} images; bit-identical per layer {r['exact']}; "
              f"predictions identical {r['pred_eq']}; golden correct {r['correct']} "
              f"(legacy {r['legacy_correct']}); {dt}s")
    write_results_csv(CSV_PATH, rows, CSV_FIELDS)
    print(f"wrote {CSV_PATH} ({len(rows)} rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
