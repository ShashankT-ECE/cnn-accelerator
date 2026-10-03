"""V3 baseline package: shared helpers (numpy only; runs on the KV260 and on the laptop).

Adapted from v2/board/board_common.py @ 28dd2ad (metadata columns, env_meta, output-dir rules,
write_csv, board_id_default) -- V3 does not import v2/ at runtime.

  preprocess(x_u8, spec)       CIFAR-10 HWC uint8 -> float32 NCHW (u8/255, optional per-channel
                               mean/std); the ONE preprocessing used by the DPU export, the ONNX
                               export/calibration and the board session
  Package / load_package       baseline package (make_baseline_package.py): MANIFEST.json sha256 check
  meta / env_meta / write_csv  results rows (v3/model/common.py META_COLUMNS first, then EXTRA_META)
  check_output_path            dry-run output only under a 'dryrun' directory, hardware output never

Layouts: in the repo this file is v3/board/ (results -> v3/results/, dry runs -> v3/results/dryrun/);
on the board make_baseline_package.py + deploy_baseline.sh put it in ~/gos3/ (results -> ~/gos3/results/).
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOARD_DIR = Path(__file__).resolve().parent
IN_REPO = (BOARD_DIR.parent / "model" / "nets.py").is_file()
REPO_ROOT = BOARD_DIR.parents[1] if IN_REPO else None
RESULTS_ROOT = (BOARD_DIR.parent / "results") if IN_REPO else (BOARD_DIR / "results")
PACKAGE_DIR_REPO = BOARD_DIR / "data" / "package"          # gitignored (v3/.gitignore board/data/)
MANIFEST = "MANIFEST.json"

# = v3/model/common.py META_COLUMNS (copied: the board has no v3/model)
META_COLUMNS = (
    "timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256",
    "board_id", "net", "layer", "clock_mhz", "source", "duration_s", "num_inferences",
)
EXTRA_META = ("board_hostname", "clock_source", "scripts_commit", "package_manifest_sha256",
              "checkpoint_sha256", "checkpoint_kind", "backend",
              # measurement environment (baseline_session.py pre-flight)
              "session_index", "paper_grade", "env_step", "cpu_governor", "cpu_freq_khz",
              "cpu_affinity", "die_temp_start_c", "env_note")
SOURCE_HW = "measured on KV260"
SOURCE_DRYRUN = "dryrun_model"
CHECKPOINT_TRAINED = "external"              # a checkpoint passed in by path (GPU training job)
CHECKPOINT_THROWAWAY = "dryrun_throwaway"    # v3/dpu/make_dryrun_ckpt.py: never paper data

# CIFAR-10 preprocessing. The V2 convention (u8/255, no mean/std) is the default ONLY because the
# V3 training recipe is not committed yet: the checkpoint's own "preprocess" entry (or the CLI) wins.
# OPEN (V3 DECISIONS): must equal the v3/train preprocessing of the checkpoint.
DEFAULT_PREPROCESS = {"scale": "1/255", "mean": None, "std": None,
                      "source": "default (V2 convention: u8/255, no mean/std); OPEN: must match v3/train"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---- preprocessing ---------------------------------------------------------------------------
def preprocess(x_u8: np.ndarray, spec: dict | None = None) -> np.ndarray:
    """CIFAR-10 uint8 HWC ([N,32,32,3] or [32,32,3]) -> float32 NCHW. u8 / 255 in float32 (= torchvision
    ToTensor), then, if spec has mean/std, (x - mean) / std per channel in float32 (= Normalize)."""
    spec = spec or DEFAULT_PREPROCESS
    x = np.asarray(x_u8)
    assert x.dtype == np.uint8, x.dtype
    single = x.ndim == 3
    if single:
        x = x[None]
    assert x.shape[1:] == (32, 32, 3), x.shape
    f = np.ascontiguousarray(x.transpose(0, 3, 1, 2)).astype(np.float32) / np.float32(255.0)
    if spec.get("mean") is not None:
        m = np.asarray(spec["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        s = np.asarray(spec["std"], dtype=np.float32).reshape(1, -1, 1, 1)
        f = (f - m) / s
    return f[0] if single else f


def preprocess_hwc(x_u8: np.ndarray, spec: dict | None = None) -> np.ndarray:
    """Same arithmetic as preprocess() but kept in HWC layout (the DPU input layout): elementwise
    identical values, so preprocess_hwc(x) == preprocess(x).transpose(...) bit for bit."""
    spec = spec or DEFAULT_PREPROCESS
    x = np.asarray(x_u8)
    assert x.dtype == np.uint8, x.dtype
    f = x.astype(np.float32) / np.float32(255.0)
    if spec.get("mean") is not None:
        f = (f - np.asarray(spec["mean"], dtype=np.float32)) / np.asarray(spec["std"], dtype=np.float32)
    return f


def to_nhwc(x_nchw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(x_nchw, (0, 2, 3, 1)))


# ---- package ---------------------------------------------------------------------------------
class ManifestError(RuntimeError):
    pass


def verify_manifest(pkg_dir: Path) -> dict:
    """Every file listed in MANIFEST.json: size + sha256. Returns the manifest."""
    pkg_dir = Path(pkg_dir)
    mp = pkg_dir / MANIFEST
    if not mp.is_file():
        raise ManifestError(f"{mp} missing (build with make_baseline_package.py, deploy with deploy_baseline.sh)")
    man = json.loads(mp.read_text())
    for name, meta in man["files"].items():
        p = pkg_dir / name
        if not p.is_file():
            raise ManifestError(f"{p} missing")
        if p.stat().st_size != meta["bytes"]:
            raise ManifestError(f"{p}: size {p.stat().st_size} != manifest {meta['bytes']}")
        if sha256_file(p) != meta["sha256"]:
            raise ManifestError(f"{p}: sha256 differs from {MANIFEST}")
    return man


@dataclass
class Package:
    dir: Path
    manifest: dict
    manifest_sha256: str
    _test: dict = field(default_factory=dict)

    @property
    def nets(self) -> list[str]:
        return list(self.manifest["nets"])

    def net(self, net: str) -> dict:
        return self.manifest["nets"][net]

    def _load_test(self):
        if not self._test:
            z = np.load(self.dir / self.manifest["test_set"]["file"])
            self._test.update(x_u8=z["x_u8"], y=z["y"].astype(np.int64))
        return self._test

    @property
    def x_u8(self) -> np.ndarray:
        return self._load_test()["x_u8"]

    @property
    def labels(self) -> np.ndarray:
        return self._load_test()["y"]

    def preprocess_spec(self, net: str) -> dict:
        return self.net(net)["preprocess"]

    def path(self, rel: str) -> Path:
        return self.dir / rel

    @property
    def dirty(self) -> bool:
        return bool(self.manifest.get("git_dirty", True))

    def checkpoint_kind(self, net: str) -> str:
        return self.net(net).get("checkpoint_kind", "")


def load_package(pkg_dir, verify: bool = True) -> Package:
    pkg_dir = Path(pkg_dir)
    man = verify_manifest(pkg_dir) if verify else json.loads((pkg_dir / MANIFEST).read_text())
    return Package(pkg_dir, man, sha256_file(pkg_dir / MANIFEST))


# ---- provenance / rows -----------------------------------------------------------------------
def board_id_default() -> str:
    """= v2 board_common.board_id_default: GOS_BOARD_ID, else device-tree model + machine-id prefix."""
    env = os.environ.get("GOS_BOARD_ID")
    if env:
        return env
    try:
        model = Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode()
    except OSError:
        model = platform.machine()
    try:
        mid = Path("/etc/machine-id").read_text().strip()[:8]
    except OSError:
        mid = "nomachineid"
    return f"{model} machine-id:{mid}"


def env_meta(source: str, dirty: bool, env: dict | None, checkpoint_kind: str = CHECKPOINT_TRAINED) -> dict:
    """Environment columns. paper_grade only for a hardware row from a clean tree, a trained (not
    throwaway) checkpoint, and a step that passed the environment pre-flight."""
    env = env or {}
    try:
        aff = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        aff = ""
    pg = (bool(env.get("paper_grade")) and source == SOURCE_HW and not dirty
          and checkpoint_kind == CHECKPOINT_TRAINED)
    if not env:
        note = "no environment pre-flight: not paper-grade"
    elif source != SOURCE_HW:
        note = "DRY RUN environment (fake sysfs, laptop ORT, fake DPU runner, mock sensor), not data"
    elif checkpoint_kind != CHECKPOINT_TRAINED:
        note = f"checkpoint kind {checkpoint_kind}: not paper-grade"
    else:
        note = env.get("note", "")
    return {"session_index": env.get("session_index", ""), "paper_grade": pg,
            "env_step": env.get("step_id", ""), "cpu_governor": env.get("cpu_governor", ""),
            "cpu_freq_khz": env.get("cpu_freq_khz", ""), "cpu_affinity": aff,
            "die_temp_start_c": env.get("die_temp_start_c", ""), "env_note": note}


def resolve_out_dir(source: str, out_dir) -> Path:
    out = Path(out_dir) if out_dir is not None else (
        RESULTS_ROOT / "dryrun" / "baseline" if source == SOURCE_DRYRUN else RESULTS_ROOT)
    check_output_path(out / "x.csv", source)
    out.mkdir(parents=True, exist_ok=True)
    return out


def check_output_path(path: Path, source: str):
    """Dry-run output only under a 'dryrun' directory; hardware output never there."""
    parts = Path(os.path.abspath(path)).parent.parts
    in_dryrun = "dryrun" in parts
    if source == SOURCE_DRYRUN and not in_dryrun:
        raise SystemExit(f"REFUSED: dry-run ({SOURCE_DRYRUN}) output must go under a 'dryrun' directory, "
                         f"not {path}")
    if source != SOURCE_DRYRUN and in_dryrun:
        raise SystemExit(f"REFUSED: hardware output must not go under a 'dryrun' directory: {path}")


def write_csv(path, rows: list[dict], fields: list[str], source: str) -> Path:
    path = Path(path)
    check_output_path(path, source)
    for r in rows:
        if r.get("source") != source:
            raise SystemExit(f"REFUSED: row source {r.get('source')!r} != run source {source!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(META_COLUMNS) + list(EXTRA_META) + [f for f in fields if f not in META_COLUMNS + EXTRA_META]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {path} ({len(rows)} rows, source={source})")
    return path


def read_csv(path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))
