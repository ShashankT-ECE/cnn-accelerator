"""Network parameter configuration for v2/model (single switch point).

Every V2 script loads network parameters through ``NET_CONFIGS``. Switching a
net to a new parameter set is a one-line change of its ``param_dir`` below.

Step 2.1c (DECISIONS D3/D9): ``cifar10`` is the retrained r2 set
(``frozen/cifar10_int8_r2``). The original set is kept as ``cifar10_r1`` for the
record only: it is not in ``NETS`` (the nets the hardware runs, vectors, cycle
model), and is used only by the accuracy-of-record and r1 provenance checks.

Each param_dir holds (or, for LeNet-5, points to) ``quant_params.npz`` (legacy
key convention: ``{layer}_q_w`` int8, ``{layer}_q_b`` int32, ``{layer}_S_*`` /
``{layer}_M`` float64, ``S_input``) and ``hw_requant.npz`` (Step 2.1:
``{layer}_m`` uint64, ``{layer}_s`` uint8, ``B``).
"""
from __future__ import annotations

from pathlib import Path

from common import FROZEN_DIR, REPO_ROOT

# ---- the switch: one line per net -------------------------------------------
LENET5_PARAM_DIR = FROZEN_DIR / "lenet5_int8"        # POINTER.md -> data/lenet5_int8/
CIFAR10_PARAM_DIR = FROZEN_DIR / "cifar10_int8_r2"   # D3 (Step 2.1c): r2 adopted
CIFAR10_R1_PARAM_DIR = FROZEN_DIR / "cifar10_int8"   # r1, history only (not in NETS)
# ------------------------------------------------------------------------------


def _layer(name, IC, OC, KH, KW, IH, IW, relu, pool, final=False):
    OH, OW = IH - KH + 1, IW - KW + 1
    return {"name": name, "IC": IC, "OC": OC, "KH": KH, "KW": KW, "IH": IH, "IW": IW,
            "OH": OH, "OW": OW, "relu": relu, "pool": pool, "final": final,
            "K": IC * KH * KW}


# Layer lists per ARCH_SPEC "Workloads" (VALID, stride 1; FC = 1x1 conv on a 1x1 map).
# Names are the legacy parameter keys.
LENET5_LAYERS = (
    _layer("conv1", 1, 6, 5, 5, 32, 32, relu=True, pool=True),
    _layer("conv3", 6, 16, 5, 5, 14, 14, relu=True, pool=True),
    _layer("conv5", 16, 120, 5, 5, 5, 5, relu=True, pool=False),
    _layer("fc1", 120, 84, 1, 1, 1, 1, relu=True, pool=False),
    _layer("fc2", 84, 10, 1, 1, 1, 1, relu=False, pool=False, final=True),
)
CIFAR10_LAYERS = (
    _layer("conv1", 3, 32, 5, 5, 32, 32, relu=True, pool=True),
    _layer("conv2", 32, 32, 5, 5, 14, 14, relu=True, pool=True),
    _layer("conv3", 32, 64, 5, 5, 5, 5, relu=True, pool=False),
    _layer("fc", 64, 10, 1, 1, 1, 1, relu=False, pool=False, final=True),
)


def _lenet5_quant_params(param_dir: Path) -> Path:
    # LeNet-5 is not copied under v2/: the frozen legacy file is the source.
    if param_dir == FROZEN_DIR / "lenet5_int8":
        return REPO_ROOT / "data" / "lenet5_int8" / "quant_params.npz"
    return param_dir / "quant_params.npz"


NET_CONFIGS = {
    "lenet5": {
        "param_dir": LENET5_PARAM_DIR,
        "quant_params": _lenet5_quant_params(LENET5_PARAM_DIR),
        "hw_requant": LENET5_PARAM_DIR / "hw_requant.npz",
        "layers": LENET5_LAYERS,
        "dataset": "mnist",
        "checkpoint": "data/checkpoint/lenet5_fp32.pt",
        "reference_version": "lenet5_v1",
    },
    "cifar10": {
        "param_dir": CIFAR10_PARAM_DIR,
        "quant_params": CIFAR10_PARAM_DIR / "quant_params.npz",
        "hw_requant": CIFAR10_PARAM_DIR / "hw_requant.npz",
        "layers": CIFAR10_LAYERS,
        "dataset": "cifar10",
        "checkpoint": "v2/model/retrain/cifar10_fp32_r2.pt",
        "reference_version": "cifar10_r2",
    },
    "cifar10_r1": {
        "param_dir": CIFAR10_R1_PARAM_DIR,
        "quant_params": CIFAR10_R1_PARAM_DIR / "quant_params.npz",
        "hw_requant": CIFAR10_R1_PARAM_DIR / "hw_requant.npz",
        "layers": CIFAR10_LAYERS,
        "dataset": "cifar10",
        "checkpoint": "data/checkpoint/cifar10_fp32.pt",
        "reference_version": "cifar10_r1",
    },
}
# Nets the accelerator runs (vectors, cycle model, golden). cifar10_r1 is record-only.
NETS = ("lenet5", "cifar10")
RECORD_ONLY = ("cifar10_r1",)
assert set(NETS) | set(RECORD_ONLY) == set(NET_CONFIGS)


def layer_by_name(net: str, name: str) -> dict:
    return next(L for L in NET_CONFIGS[net]["layers"] if L["name"] == name)
