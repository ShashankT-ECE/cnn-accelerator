"""V2 final layer: raw INT32 logits -> reference float32 logits -> argmax (D2).

ARCH_SPEC "Final layer": the hardware does no requant on the final FC layer; it
exports raw INT32 ``v = acc + q_bias`` per class and the PS applies the
reference's own float32 dequantization and argmax.

The legacy reference has no callable for this step; the dequant is inline in
``forward_layers``:

* LeNet-5  : python/lenet5/int8_model.py:177-181 (fc2)
      acc = r.astype(np.int64) @ L["q_w"].astype(np.int64).T
      logits = (acc + L["q_b"]) * (L["S_a"] * L["S_w"]).reshape(1, -1)  # float64
      out["logits"] = logits.astype(np.float32)
  argmax: python/eval_lenet5_int8.py:54  ``model.forward(x).argmax(1)``
* CIFAR-10 : python/cifar10/int8_model.py:111-115 (fc), same expression
  argmax: python/eval_cifar10_int8.py:47 ``model.forward(x).argmax(1)``

``logits_from_raw`` reproduces that expression with the same dtypes and
operation order: int64 (acc int64 + q_b int32 -> int64) times the float64
per-channel scale ``S_a * S_w`` -> float64, then cast to float32. Converting v
through int32 is lossless as long as |v| < 2^31 (checked by ``check_net``).
(python/reference/quant.py:90 ``dequantize_int32`` computes the same product
but broadcasts channel-first, and forward_layers does not use it.)

Run ``python final_layer.py`` from v2/model to write
v2/results/final_layer_check.csv (source = model).
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from common import REPO_ROOT, RESULTS_DIR, base_meta, write_results_csv

CALIB_N = 1024          # train[0:1024], as eval_*_int8.py
TEST_N = 10000
INT8_MAG = 128          # max |int8|
INT32_MAX = 2**31 - 1

# net -> (final layer name, forward_layers acc key, K of final layer)
NETS = {
    "lenet5": ("fc2", "out_acc", 84),
    "cifar10": ("fc", "fc_acc", 64),
}

CSV_FIELDS = ["images", "argmax_match", "logits_bitexact", "max_abs_v",
              "bound_abs_v", "int32_fits", "raw_argmax_differs"]


def logits_from_raw(v_int32: np.ndarray, params: dict) -> np.ndarray:
    """Reference float32 logits from raw INT32 ``v = acc + q_bias`` (N, C).

    ``params`` is the legacy final-layer dict (uses ``S_a``, ``S_w``).
    Mirrors python/lenet5/int8_model.py:180-181 / python/cifar10/int8_model.py:114-115.
    """
    v = np.asarray(v_int32)
    assert v.dtype == np.int32, v.dtype
    scale = (params["S_a"] * params["S_w"]).reshape(1, -1)   # float64 (1, C)
    logits = v.astype(np.int64) * scale                      # float64 (N, C)
    return logits.astype(np.float32)


def predict_from_raw(v_int32: np.ndarray, params: dict) -> np.ndarray:
    """Class prediction exactly as the legacy eval: float32 logits .argmax(1)."""
    return logits_from_raw(v_int32, params).argmax(1)


def raw_to_int32(acc: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """v = acc + q_b (int64 as in legacy), asserted to fit int32, cast to int32."""
    v64 = acc.astype(np.int64) + q_b.astype(np.int64).reshape(1, -1)
    assert np.abs(v64).max() <= INT32_MAX, "final-layer v does not fit int32"
    v32 = v64.astype(np.int32)
    assert np.array_equal(v32.astype(np.int64), v64)
    return v32


# --------------------------------------------------------------------------
# Legacy model / data loading (legacy code called, not re-implemented)
# --------------------------------------------------------------------------

def _legacy(net: str):
    from torchvision import datasets
    if net == "lenet5":
        from lenet5.int8_model import Int8LeNet5 as Model, calibrate, load_weights
        from lenet5.preprocess import make_transform
        ds_cls, ckpt = datasets.MNIST, REPO_ROOT / "data/checkpoint/lenet5_fp32.pt"
    elif net == "cifar10":
        from cifar10.int8_model import Int8Cifar10Net as Model, calibrate, load_weights
        from cifar10.preprocess import make_transform
        ds_cls, ckpt = datasets.CIFAR10, REPO_ROOT / "data/checkpoint/cifar10_fp32.pt"
    else:
        raise ValueError(net)
    return Model, calibrate, load_weights, make_transform, ds_cls, ckpt


def _dataset(net: str, train: bool):
    _, _, _, make_transform, ds_cls, _ = _legacy(net)
    return ds_cls(root=str(REPO_ROOT / "data/raw"), train=train, download=False,
                  transform=make_transform())


@lru_cache(maxsize=None)
def calibrated_params(net: str) -> dict:
    """Legacy calibrate(load_weights(ckpt), train[0:1024]) as eval_*_int8.py."""
    _, calibrate, load_weights, _, _, ckpt = _legacy(net)
    tr = _dataset(net, train=True)
    calib = np.stack([tr[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)
    return calibrate(load_weights(ckpt), calib)


@lru_cache(maxsize=None)
def test_images(net: str, n: int = TEST_N) -> np.ndarray:
    te = _dataset(net, train=False)
    return np.stack([te[i][0].numpy() for i in range(n)]).astype(np.float32)


def legacy_model(net: str):
    return _legacy(net)[0](calibrated_params(net))


# --------------------------------------------------------------------------
# Equivalence check
# --------------------------------------------------------------------------

def check_net(net: str, n: int = TEST_N, batch: int = 500) -> dict:
    """Compare logits_from_raw(v) against legacy forward_layers on n test images."""
    layer, acc_key, K = NETS[net]
    model = legacy_model(net)
    P = calibrated_params(net)[layer]
    x_all = test_images(net, TEST_N)[:n]

    argmax_match = bitexact = raw_diff = 0
    max_abs_v = 0
    for s in range(0, n, batch):
        out = model.forward_layers(x_all[s:s + batch])
        ref_logits = out["logits"]                         # legacy float32
        ref_pred = ref_logits.argmax(1)                    # legacy prediction
        v = raw_to_int32(out[acc_key], P["q_b"])
        max_abs_v = max(max_abs_v, int(np.abs(v.astype(np.int64)).max()))
        got = logits_from_raw(v, P)
        assert got.dtype == np.float32
        row_bits = np.all(got.view(np.uint32) == ref_logits.view(np.uint32), axis=1)
        row_eq = np.all(got == ref_logits, axis=1)
        assert np.array_equal(row_bits, row_eq)
        bitexact += int(row_bits.sum())
        argmax_match += int((got.argmax(1) == ref_pred).sum())
        raw_diff += int((v.argmax(1) != ref_pred).sum())

    bound = K * INT8_MAG * INT8_MAG + int(np.abs(P["q_b"].astype(np.int64)).max())
    return {
        "net": net, "layer": layer, "images": n,
        "argmax_match": argmax_match, "logits_bitexact": bitexact,
        "max_abs_v": max_abs_v, "bound_abs_v": bound,
        "int32_fits": bool(max_abs_v <= INT32_MAX and bound <= INT32_MAX),
        "raw_argmax_differs": raw_diff,
    }


def main() -> int:
    rows = []
    for net in NETS:
        t0 = time.time()
        r = check_net(net)
        dur = round(time.time() - t0, 2)
        print(r)
        meta = base_meta(net=net, layer=r["layer"], source="model",
                         duration_s=dur, num_inferences=r["images"])
        rows.append({**meta, **{k: r[k] for k in CSV_FIELDS}})
    p = write_results_csv(RESULTS_DIR / "final_layer_check.csv", rows, CSV_FIELDS)
    print(f"wrote {p}")
    ok = all(r["argmax_match"] == r["images"] == r["logits_bitexact"] and r["int32_fits"]
             for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
