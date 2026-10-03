"""V3 baselines, LAPTOP ONLY (repo .venv, torch): CIFAR-10 raw data, checkpoints, git state.

Used by v3/dpu/export_data.py, v3/dpu/make_dryrun_ckpt.py, v3/board/export_onnx.py and
v3/board/make_baseline_package.py. Never deployed to the board.

CIFAR-10 is read from the python pickles in data/raw/cifar-10-batches-py (the files torchvision
reads; same order: test_batch = test[0:10000], data_batch_1..5 = train[0:50000]), as uint8 HWC.

Checkpoints (external GPU training writes them later to v3/train/runs/<job>/...pt; the format is not
fixed yet) are accepted as
  * a plain state_dict, or
  * a dict with one of the keys state_dict / model_state_dict / model (-> the state_dict) and optional
    "net" (must equal --net), "preprocess" ({"mean": [3], "std": [3]}, see baseline_common.preprocess),
    "dryrun_throwaway" (True only for make_dryrun_ckpt.py output).
A "module." prefix (DataParallel) is stripped. The model is v3/model/nets.build(net) in eval mode.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

BOARD = Path(__file__).resolve().parent
V3 = BOARD.parent
REPO = V3.parent
sys.path.insert(0, str(BOARD))
sys.path.insert(0, str(V3 / "model"))

import baseline_common as bc  # noqa: E402

CIFAR_DIR = REPO / "data" / "raw" / "cifar-10-batches-py"
TEST_N = 10000
CALIB_N = 1024
# Generated outputs excluded from the dirty check (= v3/model/common.OUTPUT_PATHSPECS).
OUTPUT_PATHSPECS = (":!v3/results",)


def git_state() -> dict:
    g = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,  # noqa: E731
                                  text=True, check=True).stdout.strip()
    return {"git_commit": g("rev-parse", "HEAD"),
            "git_dirty": bool(g("status", "--porcelain", "--", ".", *OUTPUT_PATHSPECS))}


def _batch(name: str):
    with (CIFAR_DIR / name).open("rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.asarray(d[b"labels"], dtype=np.int64)


def cifar_test() -> tuple[np.ndarray, np.ndarray]:
    """test[0:10000]: uint8 [10000,32,32,3] HWC, int64 labels."""
    return _batch("test_batch")


def cifar_train_head(n: int = CALIB_N) -> tuple[np.ndarray, np.ndarray]:
    """train[0:n] (data_batch_1 first, as torchvision)."""
    xs, ys, k = [], [], 1
    while sum(len(y) for y in ys) < n:
        x, y = _batch(f"data_batch_{k}")
        xs.append(x)
        ys.append(y)
        k += 1
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]


def load_checkpoint(net: str, path, preprocess_override: dict | None = None):
    """(model eval, info) for v3/model/nets.build(net) + the checkpoint at path. info: checkpoint
    path / sha256 / kind, the state_dict (cleaned), preprocess spec."""
    import torch
    import nets
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    meta = {}
    if isinstance(obj, dict) and any(k in obj for k in ("state_dict", "model_state_dict", "model")):
        meta = obj
        sd = next(obj[k] for k in ("state_dict", "model_state_dict", "model") if k in obj)
    else:
        sd = obj
    if not isinstance(sd, dict):
        raise SystemExit(f"{path}: no state_dict found (got {type(sd).__name__})")
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    if meta.get("net") not in (None, net):
        raise SystemExit(f"{path}: checkpoint net {meta.get('net')!r} != --net {net!r}")
    model = nets.build(net)
    model.load_state_dict(sd, strict=True)
    model.eval()
    if preprocess_override:
        pp = dict(preprocess_override, scale="1/255", source="command line")
    elif isinstance(meta.get("preprocess"), dict) and meta["preprocess"].get("mean") is not None:
        pp = {"scale": "1/255", "mean": [float(v) for v in meta["preprocess"]["mean"]],
              "std": [float(v) for v in meta["preprocess"]["std"]], "source": "checkpoint 'preprocess' entry"}
    else:
        pp = dict(bc.DEFAULT_PREPROCESS)
    kind = bc.CHECKPOINT_THROWAWAY if meta.get("dryrun_throwaway") else bc.CHECKPOINT_TRAINED
    info = {"checkpoint": str(path.resolve().relative_to(REPO)) if path.resolve().is_relative_to(REPO)
            else str(path.resolve()), "checkpoint_sha256": bc.sha256_file(path), "checkpoint_kind": kind,
            "preprocess": pp, "state_dict": model.state_dict(), "load": "strict OK"}
    return model, info


def predict_torch(model, x: np.ndarray, batch: int = 500) -> tuple[np.ndarray, np.ndarray]:
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            out.append(model(torch.from_numpy(x[s:s + batch])).float().numpy())
    lg = np.concatenate(out).astype(np.float32)
    return lg, lg.argmax(1).astype(np.int64)


def parse_preprocess(mean, std) -> dict | None:
    if mean is None and std is None:
        return None
    if mean is None or std is None or len(mean) != 3 or len(std) != 3:
        raise SystemExit("--mean and --std need 3 values each")
    return {"mean": [float(v) for v in mean], "std": [float(v) for v in std]}
