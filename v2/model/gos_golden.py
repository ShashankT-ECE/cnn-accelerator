"""V2 step 2.2 part A: integer-only golden model of the gos_ accelerator.

Implements ARCH_SPEC "Workloads" + "Numeric contract" per layer:

    acc = VALID conv, stride 1, int8 x int8 -> int64 accumulation (asserted to
          fit INT32, the hardware accumulator width)
    v   = acc + q_bias
    non-final layer:  q = hw_requant(v, m[c], s[c])  (exact integer RNE shift,
                      clip [-128, 127]; per output channel, requant_check.py),
                      ReLU if relu, 2x2/2 max pool if pool (on int8 after ReLU)
    final layer:      raw v returned as INT32 (asserted to fit), no requant

FC layers are 1x1 convs on a 1x1 map: FC weights [OC, IC] are reshaped to
[OC, IC, 1, 1] and the input of an FC layer is the previous layer's
[C, 1, 1] output map. Because that map is 1x1, the legacy flatten
``r.reshape(N, -1)`` of an (N, C, 1, 1) tensor is exactly channel order c, so
no flatten reorder is needed (verified bit-exact in tests/test_gos_golden.py).

The layer path (``gos_conv_acc`` / ``gos_layer``) uses integer numpy
operations only; dtypes are asserted (inputs int8, acc/v int64, outputs int8
or int32). Two steps are deliberately *outside* the integer path and run on
the host / PS, not in the accelerator:

* ``quantize_input``: float input -> int8 with the legacy
  ``reference.quant.quantize_tensor(x, S_input)`` (float64 RNE) — the host
  quantizes the image before it is written to the ACT buffer.
* the prediction: raw INT32 logits -> legacy float32 dequant + argmax via
  ``final_layer.predict_from_raw`` (DECISIONS D2).

All parameters are loaded through ``net_config.NET_CONFIGS`` only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import common  # noqa: F401  (puts legacy python/ on sys.path)
from final_layer import logits_from_raw
from net_config import NET_CONFIGS
from requant_check import hw_requant

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class LayerParams:
    cfg: dict                          # the NET_CONFIGS layer dict
    q_w: np.ndarray                    # int8 [OC, IC, KH, KW]
    q_b: np.ndarray                    # int32 [OC]
    m: tuple = ()                      # per-channel multiplier (Python ints), non-final
    s: tuple = ()                      # per-channel shift (Python ints), non-final
    S_a: float | None = None           # final layer only (PS dequant)
    S_w: np.ndarray | None = None      # final layer only (PS dequant)

    @property
    def name(self) -> str:
        return self.cfg["name"]


@dataclass
class GosNet:
    net: str
    S_input: float
    B: int
    layers: list = field(default_factory=list)   # list[LayerParams]

    @property
    def final(self) -> LayerParams:
        return self.layers[-1]


def load_net(net: str) -> GosNet:
    """Per-layer parameters from NET_CONFIGS[net] (quant_params + hw_requant)."""
    cfg = NET_CONFIGS[net]
    qp = np.load(cfg["quant_params"])
    hw = np.load(cfg["hw_requant"])
    B = int(hw["B"])
    layers = []
    for i, L in enumerate(cfg["layers"]):
        n = L["name"]
        assert L["final"] == (i == len(cfg["layers"]) - 1), f"{net}: final must be last"
        q_w = qp[f"{n}_q_w"]
        q_b = qp[f"{n}_q_b"]
        assert q_w.dtype == np.int8 and q_b.dtype == np.int32, (n, q_w.dtype, q_b.dtype)
        shape = (L["OC"], L["IC"], L["KH"], L["KW"])
        if q_w.ndim == 2:                                  # FC [OC, IC] -> 1x1 conv
            assert L["KH"] == L["KW"] == 1, n
            q_w = q_w.reshape(q_w.shape[0], q_w.shape[1], 1, 1)
        assert q_w.shape == shape, (n, q_w.shape, shape)
        assert q_b.shape == (L["OC"],), n
        assert int(np.prod(shape[1:])) == L["K"], n
        if L["final"]:
            assert f"{n}_m" not in hw.files, f"{n}: final layer must have no requant"
            P = LayerParams(L, q_w, q_b, S_a=float(qp[f"{n}_S_a"]),
                            S_w=np.asarray(qp[f"{n}_S_w"], dtype=np.float64))
            assert P.S_w.shape == (L["OC"],)
        else:
            m, s = hw[f"{n}_m"], hw[f"{n}_s"]
            assert m.shape == s.shape == (L["OC"],), n
            m = tuple(int(x) for x in m)
            s = tuple(int(x) for x in s)
            assert all((1 << (B - 1)) <= x < (1 << B) for x in m), f"{n}: m not B={B} bits"
            assert all(1 <= x <= 63 for x in s), n
            P = LayerParams(L, q_w, q_b, m=m, s=s)
        layers.append(P)
    return GosNet(net=net, S_input=float(qp["S_input"]), B=B, layers=layers)


# --------------------------------------------------------------------------- #
# Host / PS steps (float, outside the integer layer path)
# --------------------------------------------------------------------------- #
def quantize_input(net, x_fp: np.ndarray) -> np.ndarray:
    """Host step: float image(s) -> int8 with the legacy per-tensor quantizer.

    ``reference.quant.quantize_tensor(x, S_input).astype(int8)`` exactly as the
    legacy ``forward_layers`` (float64 RNE, clip). Not part of the integer path.
    """
    from reference import quant
    G = net if isinstance(net, GosNet) else load_net(net)
    return quant.quantize_tensor(x_fp, G.S_input).astype(np.int8)


def _final_params(P: LayerParams) -> dict:
    return {"S_a": P.S_a, "S_w": P.S_w}


# --------------------------------------------------------------------------- #
# Integer layer path
# --------------------------------------------------------------------------- #
def gos_conv_acc(x: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    """VALID stride-1 conv: int8 [N,C,H,W] x int8 [OC,C,KH,KW] -> int64 [N,OC,OH,OW].

    im2col over k = (ic, ky, kx) (kx innermost, as the hardware stream) and an
    int64 einsum (integer arithmetic, exact).
    """
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
    """2x2 stride-2 max pool on int8 [N,C,H,W] (H, W even)."""
    assert x.dtype == np.int8
    N, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0, (H, W)
    return x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))


def gos_layer(x: np.ndarray, L: dict, P: LayerParams, return_v: bool = False):
    """One layer. x int8 [C,H,W] or [N,C,H,W].

    Non-final: returns int8 [.., OC, OH', OW'] (after requant, ReLU, pool).
    Final: returns raw INT32 v [.., OC, OH, OW].
    With ``return_v`` also returns v (int64) = acc + q_bias.
    """
    assert x.dtype == np.int8, x.dtype
    single = x.ndim == 3
    if single:
        x = x[None]
    assert x.shape[1:] == (L["IC"], L["IH"], L["IW"]), (L["name"], x.shape)
    acc = gos_conv_acc(x, P.q_w)
    assert acc.shape[1:] == (L["OC"], L["OH"], L["OW"])
    assert acc.size == 0 or (INT32_MIN <= acc.min() and acc.max() <= INT32_MAX), \
        f"{L['name']}: acc exceeds INT32"
    v = acc + P.q_b.astype(np.int64).reshape(1, -1, 1, 1)
    assert v.dtype == np.int64

    if L["final"]:
        assert v.size == 0 or (INT32_MIN <= v.min() and v.max() <= INT32_MAX), \
            f"{L['name']}: final v exceeds INT32"
        y = v.astype(np.int32)
    else:
        y = np.empty(v.shape, dtype=np.int8)
        for c in range(L["OC"]):
            y[:, c] = hw_requant(v[:, c], P.m[c], P.s[c])
        assert y.dtype == np.int8
        if L["relu"]:
            y = np.maximum(y, np.int8(0))
        if L["pool"]:
            y = gos_maxpool2x2(y)
        assert y.dtype == np.int8
    if single:
        y, v = y[0], v[0]
    return (y, v) if return_v else y


def run_net(net, x_int8: np.ndarray) -> dict:
    """Run all layers on int8 input [N,C,H,W] (or [C,H,W]).

    Returns {"input": x, <layer name>: output (int8; final layer int32 v),
    "v": final raw INT32 v [N, OC], "logits": PS float32 logits,
    "pred": PS argmax}. For a single image the batch axis is dropped.
    """
    G = net if isinstance(net, GosNet) else load_net(net)
    assert x_int8.dtype == np.int8
    single = x_int8.ndim == 3
    x = x_int8[None] if single else x_int8
    out = {"input": x}
    for P in G.layers:
        x = gos_layer(x, P.cfg, P)
        out[P.name] = x
    assert x.dtype == np.int32 and x.shape[2:] == (1, 1), x.shape
    v = x.reshape(x.shape[0], -1)
    out["v"] = v
    # PS step (float, outside the accelerator): legacy float32 dequant + argmax.
    out["logits"] = logits_from_raw(v, _final_params(G.final))
    out["pred"] = out["logits"].argmax(1)
    if single:
        out = {k: a[0] for k, a in out.items()}
    return out


# --------------------------------------------------------------------------- #
# Dataset (as python/eval_{lenet5,cifar10}_int8.py)
# --------------------------------------------------------------------------- #
def load_test_set(net: str, n: int = 10000):
    """(x float32 [n,C,32,32], y int64 [n]) from torchvision, root data/raw,
    download=False, legacy make_transform() — identical to eval_*_int8.py."""
    from torchvision import datasets
    from common import REPO_ROOT
    ds_name = NET_CONFIGS[net]["dataset"]
    if ds_name == "mnist":
        from lenet5.preprocess import make_transform
        ds_cls = datasets.MNIST
    elif ds_name == "cifar10":
        from cifar10.preprocess import make_transform
        ds_cls = datasets.CIFAR10
    else:
        raise ValueError(ds_name)
    ds = ds_cls(root=str(REPO_ROOT / "data" / "raw"), train=False, download=False,
                transform=make_transform())
    items = [ds[i] for i in range(n)]
    x = np.stack([it[0].numpy() for it in items]).astype(np.float32)
    y = np.array([it[1] for it in items], dtype=np.int64)
    return x, y
