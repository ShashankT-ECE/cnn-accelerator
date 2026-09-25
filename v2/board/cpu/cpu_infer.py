"""V2 Step 5 (A5): same-board CPU baseline runners. numpy-only (+ optional onnxruntime).

This module must run on the KV260 (Cortex-A53, Ubuntu 22.04 / PYNQ) with numpy
only: no torch, no torchvision, no import of v2/model. Every parameter comes from
the CPU data package ``<data_dir>/<net>/cpu/`` written on the laptop by
``export_cpu_models.py``:

    int8_ref.npz    integer parameters (q_w, q_b, m, s per layer; final S_a, S_w; S_input)
    fp32.npz        FP32 checkpoint weights (conv/fc weight + bias, float32)
    fp32.onnx       FP32 torch model exported to ONNX (dynamic batch)
    int8_qdq.onnx   ORT static QDQ quantization of fp32.onnx (see CPU_MANIFEST.json)
    raw_test.npz    x_u8 (raw dataset uint8: MNIST [N,28,28], CIFAR [N,32,32,3] HWC), y
    CPU_MANIFEST.json  sha256 of every file, versions, calibration, laptop verification

Kinds (``make_runner(kind, net, data_dir, threads)``):

* ``cpu_int8_ref``   the V2 golden integer pipeline (``v2/model/gos_golden.py``),
  vendored numpy-only copy; bit-exact raw INT32 logits and float32 PS dequant
  (DECISIONS D2). Label: "unoptimized reference" (int64 einsum, not a tuned kernel).
* ``cpu_fp32_numpy`` FP32 reference network, im2col + float32 matmul (numpy BLAS).
* ``cpu_ort_fp32``   ONNX Runtime CPU EP, fp32.onnx.
* ``cpu_ort_int8``   ONNX Runtime CPU EP, int8_qdq.onnx (its own quantization, NOT
  the project INT8 numerics; own accuracy recorded).

Thread control: OpenBLAS/OpenMP thread counts are fixed when numpy is imported,
so callers must set OPENBLAS_NUM_THREADS / OMP_NUM_THREADS before importing
numpy (``run_cpu_baselines.py`` runs every thread config in its own subprocess).
``make_runner`` checks the environment and additionally applies threadpoolctl
limits when threadpoolctl is installed. ORT uses ``intra_op_num_threads``.

Each runner offers:
    runner.preprocess(x_raw) -> preprocessed tensor (int8 NCHW for cpu_int8_ref,
                                float32 NCHW otherwise); batch or single image
    runner.compute(x_pre)    -> float32 logits           (the "compute-only" call)
    runner.e2e(x_raw)        -> int prediction(s)        (preprocess + compute + argmax)
    runner(x_pre)            == runner.compute(x_pre)
cpu_int8_ref additionally has ``raw_v(x_int8)`` -> INT32 raw logits (for bit-exactness).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

KINDS = ("cpu_int8_ref", "cpu_fp32_numpy", "cpu_ort_fp32", "cpu_ort_int8")
NETS = ("lenet5", "cifar10")
THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
LIMB_K = 32
TWO62 = 1 << 62


class RunnerUnavailable(RuntimeError):
    """The kind cannot run here (e.g. onnxruntime not installed)."""


def cpu_dir(data_dir, net: str) -> Path:
    return Path(data_dir) / net / "cpu"


def load_manifest(data_dir, net: str) -> dict:
    p = cpu_dir(data_dir, net) / "CPU_MANIFEST.json"
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------------------------- #
# Preprocessing (numpy port of the legacy torchvision transforms)
#   LeNet-5 : ToTensor (uint8 / 255, float32) + Pad(2) zeros -> [1, 32, 32]
#   CIFAR-10: ToTensor (HWC uint8 -> CHW float32 / 255)       -> [3, 32, 32]
# Bit-equality with the legacy transform on all 10k test images is checked on
# the laptop by export_cpu_models.py (recorded in CPU_MANIFEST.json).
# --------------------------------------------------------------------------- #
def preprocess_fp32(net: str, x_u8: np.ndarray) -> np.ndarray:
    x = np.asarray(x_u8)
    assert x.dtype == np.uint8, x.dtype
    single = x.ndim == (2 if net == "lenet5" else 3)
    if single:
        x = x[None]
    if net == "lenet5":
        assert x.shape[1:] == (28, 28), x.shape
        f = np.zeros((x.shape[0], 1, 32, 32), dtype=np.float32)
        f[:, 0, 2:30, 2:30] = x.astype(np.float32) / np.float32(255.0)
    elif net == "cifar10":
        assert x.shape[1:] == (32, 32, 3), x.shape
        f = np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float32) / np.float32(255.0)
    else:
        raise ValueError(net)
    return f[0] if single else f


def quantize_input(x_fp32: np.ndarray, S_input: float) -> np.ndarray:
    """Legacy reference.quant.quantize_tensor: float64 RNE (np.rint), clip, int8."""
    q = np.rint(np.asarray(x_fp32, dtype=np.float64) / S_input)
    return np.clip(q, -128, 127).astype(np.int8)


# --------------------------------------------------------------------------- #
# cpu_int8_ref: vendored copy of v2/model/gos_golden.py + requant_check.hw_requant
# (integer numpy only; same operations, same dtypes, same order).
# --------------------------------------------------------------------------- #
def _rne_shift(p, s):
    p = np.asarray(p, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    one = np.int64(1)
    return (p + ((one << (s - one)) - one) + ((p >> s) & one)) >> s


def _hw_q_narrow(v, m: int, s: int):
    v = np.asarray(v, dtype=np.int64)
    return _rne_shift(v * np.int64(m), s)


def _hw_q_wide(v, m: int, s: int, k: int = LIMB_K):
    v = np.asarray(v, dtype=np.int64)
    k = min(k, s)
    sh = s - k
    mh, ml = m >> k, m & ((1 << k) - 1)
    vmax = int(np.abs(v).max()) if v.size else 0
    assert vmax * mh < TWO62 and vmax * ml < TWO62, "limb product overflow"
    a = v * np.int64(ml)
    a_hi = a >> np.int64(k)
    a_lo = a & np.int64((1 << k) - 1)
    c = v * np.int64(mh) + a_hi
    assert vmax * mh + (vmax * ml >> k) + 1 < TWO62
    f0 = (c >> np.int64(sh)) & np.int64(1)
    t = np.int64((1 << (s - 1)) - 1) + f0
    carry = (a_lo + t) >> np.int64(k)
    return (c + carry) >> np.int64(sh)


def _hw_q(v, m: int, s: int):
    v = np.asarray(v, dtype=np.int64)
    vmax = int(np.abs(v).max()) if v.size else 0
    if vmax * m < TWO62 and s <= 62:
        return _hw_q_narrow(v, m, s)
    return _hw_q_wide(v, m, s)


def hw_requant(v, m: int, s: int):
    return np.clip(_hw_q(v, m, s), -128, 127).astype(np.int8)


def gos_conv_acc(x: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    assert x.dtype == np.int8 and q_w.dtype == np.int8, (x.dtype, q_w.dtype)
    N, C, H, W = x.shape
    OC, IC, KH, KW = q_w.shape
    assert C == IC, (C, IC)
    OH, OW = H - KH + 1, W - KW + 1
    win = np.lib.stride_tricks.sliding_window_view(
        x.astype(np.int64), (KH, KW), axis=(2, 3))            # [N,C,OH,OW,KH,KW]
    win = win.transpose(0, 2, 3, 1, 4, 5).reshape(N, OH, OW, IC * KH * KW)
    wk = q_w.astype(np.int64).reshape(OC, IC * KH * KW)
    acc = np.einsum("nhwk,ok->nohw", win, wk)                   # [N,OC,OH,OW]
    assert acc.dtype == np.int64
    return acc


def gos_maxpool2x2(x: np.ndarray) -> np.ndarray:
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


class Int8RefRunner:
    kind = "cpu_int8_ref"
    label = "unoptimized reference"

    def __init__(self, net: str, data_dir, threads: int):
        self.net, self.threads = net, threads
        d = np.load(cpu_dir(data_dir, net) / "int8_ref.npz")
        self.layers = json.loads(str(d["layers_json"]))
        self.S_input = float(d["S_input"])
        self.P = []
        for L in self.layers:
            n = L["name"]
            p = {"q_w": d[f"{n}_q_w"], "q_b": d[f"{n}_q_b"]}
            assert p["q_w"].dtype == np.int8 and p["q_b"].dtype == np.int32
            assert p["q_w"].shape == (L["OC"], L["IC"], L["KH"], L["KW"])
            if L["final"]:
                p["scale"] = (float(d[f"{n}_S_a"]) *
                              np.asarray(d[f"{n}_S_w"], dtype=np.float64)).reshape(1, -1)
            else:
                p["m"] = tuple(int(x) for x in d[f"{n}_m"])
                p["s"] = tuple(int(x) for x in d[f"{n}_s"])
            self.P.append(p)
        self.in_shape = (self.layers[0]["IC"], self.layers[0]["IH"], self.layers[0]["IW"])

    def preprocess(self, x_u8):
        return quantize_input(preprocess_fp32(self.net, x_u8), self.S_input)

    def raw_v(self, x_int8: np.ndarray) -> np.ndarray:
        assert x_int8.dtype == np.int8
        single = x_int8.ndim == 3
        x = x_int8[None] if single else x_int8
        for L, p in zip(self.layers, self.P):
            acc = gos_conv_acc(x, p["q_w"])
            assert INT32_MIN <= acc.min() and acc.max() <= INT32_MAX, L["name"]
            v = acc + p["q_b"].astype(np.int64).reshape(1, -1, 1, 1)
            if L["final"]:
                assert INT32_MIN <= v.min() and v.max() <= INT32_MAX
                x = v.astype(np.int32)
            else:
                y = np.empty(v.shape, dtype=np.int8)
                for c in range(L["OC"]):
                    y[:, c] = hw_requant(v[:, c], p["m"][c], p["s"][c])
                if L["relu"]:
                    y = np.maximum(y, np.int8(0))
                if L["pool"]:
                    y = gos_maxpool2x2(y)
                x = y
        v = x.reshape(x.shape[0], -1)
        return v[0] if single else v

    def logits_from_raw(self, v_int32: np.ndarray) -> np.ndarray:
        """DECISIONS D2 PS step: int64 v * float64 (S_a*S_w) -> float32."""
        assert v_int32.dtype == np.int32
        single = v_int32.ndim == 1
        v = v_int32[None] if single else v_int32
        out = (v.astype(np.int64) * self.P[-1]["scale"]).astype(np.float32)
        return out[0] if single else out

    def compute(self, x_int8):
        return self.logits_from_raw(self.raw_v(x_int8))

    __call__ = compute

    def e2e(self, x_u8):
        return self.compute(self.preprocess(x_u8)).argmax(-1)


# --------------------------------------------------------------------------- #
# cpu_fp32_numpy: FP32 im2col + matmul
# --------------------------------------------------------------------------- #
class Fp32NumpyRunner:
    kind = "cpu_fp32_numpy"
    label = "numpy FP32 im2col+matmul"

    def __init__(self, net: str, data_dir, threads: int):
        self.net, self.threads = net, threads
        d = np.load(cpu_dir(data_dir, net) / "fp32.npz")
        self.layers = json.loads(str(d["layers_json"]))
        self.W, self.b = [], []
        for L in self.layers:
            w = np.asarray(d[L["name"] + "_weight"], dtype=np.float32)
            b = np.asarray(d[L["name"] + "_bias"], dtype=np.float32)
            K = L["IC"] * L["KH"] * L["KW"]
            # (K, OC) contiguous so the matmul is (P, K) @ (K, OC)
            self.W.append(np.ascontiguousarray(w.reshape(L["OC"], K).T))
            self.b.append(b)

    def preprocess(self, x_u8):
        return preprocess_fp32(self.net, x_u8)

    def compute(self, x):
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[None]
        N = x.shape[0]
        for L, W, b in zip(self.layers, self.W, self.b):
            KH, KW, IC = L["KH"], L["KW"], L["IC"]
            OH, OW = L["OH"], L["OW"]
            if KH == 1 and KW == 1 and OH == 1 and OW == 1:
                cols = x.reshape(N, IC)
            else:
                win = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(2, 3))
                # [N, IC, OH, OW, KH, KW] -> [N*OH*OW, IC*KH*KW] (k = ic, ky, kx)
                cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * OH * OW, IC * KH * KW)
            y = cols @ W + b                                   # [N*OH*OW, OC]
            if L["final"]:
                x = y.reshape(N, L["OC"])
                break
            y = y.reshape(N, OH, OW, L["OC"]).transpose(0, 3, 1, 2)
            if L["relu"]:
                y = np.maximum(y, np.float32(0))
            if L["pool"]:
                y = y.reshape(N, L["OC"], OH // 2, 2, OW // 2, 2).max(axis=(3, 5))
            x = np.ascontiguousarray(y)
        return x[0] if single else x

    __call__ = compute

    def e2e(self, x_u8):
        return self.compute(self.preprocess(x_u8)).argmax(-1)


# --------------------------------------------------------------------------- #
# ONNX Runtime
# --------------------------------------------------------------------------- #
def ort_module():
    """onnxruntime module, or None if not importable here."""
    try:
        import onnxruntime  # noqa: F401
        return onnxruntime
    except Exception:
        return None


class OrtRunner:
    label = "onnxruntime CPU EP"

    def __init__(self, kind: str, net: str, data_dir, threads: int):
        ort = ort_module()
        if ort is None:
            raise RunnerUnavailable("onnxruntime not installed")
        self.kind, self.net, self.threads = kind, net, threads
        fname = "fp32.onnx" if kind == "cpu_ort_fp32" else "int8_qdq.onnx"
        path = cpu_dir(data_dir, net) / fname
        if not path.exists():
            raise RunnerUnavailable(f"{path} missing")
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.version = ort.__version__

    def preprocess(self, x_u8):
        return preprocess_fp32(self.net, x_u8)

    def compute(self, x):
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[None]
        y = self.sess.run(None, {self.in_name: x})[0]
        return y[0] if single else y

    __call__ = compute

    def e2e(self, x_u8):
        return self.compute(self.preprocess(x_u8)).argmax(-1)


# --------------------------------------------------------------------------- #
def env_threads():
    """Thread count requested through the environment (None if inconsistent/unset)."""
    vals = {os.environ.get(k) for k in THREAD_ENV[:2]}
    return int(vals.pop()) if len(vals) == 1 and None not in vals else None


def make_runner(kind: str, net: str, data_dir, threads: int):
    """Callable runner (x_pre -> float32 logits) with .preprocess / .compute / .e2e.

    For numpy kinds, OPENBLAS_NUM_THREADS/OMP_NUM_THREADS must equal ``threads``
    (set before numpy import); threadpoolctl limits are applied too if available.
    Raises RunnerUnavailable when the kind cannot run on this machine.
    """
    if net not in NETS:
        raise ValueError(net)
    if kind in ("cpu_int8_ref", "cpu_fp32_numpy"):
        et = env_threads()
        if et != threads:
            raise RuntimeError(f"{kind}: set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS={threads} "
                               f"before importing numpy (env gives {et})")
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=threads)
        except Exception:
            pass
        cls = Int8RefRunner if kind == "cpu_int8_ref" else Fp32NumpyRunner
        return cls(net, data_dir, threads)
    if kind in ("cpu_ort_fp32", "cpu_ort_int8"):
        return OrtRunner(kind, net, data_dir, threads)
    raise ValueError(kind)
