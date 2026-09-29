"""A3-general random-shape set loader (numpy only; deployable to the KV260).

The set is produced on the laptop by v2/shapes/gen_shapes.py into v2/build/shapes/<seed>/:
``shapes.json`` (manifest) + ``jobs/J0000.npz`` ... . This module has no torch / v2.model
dependency and is safe to import on the board.

API
  verify_manifest(dir) -> dict      check format, every listed file's sha256 + size, that every
                                     jobs/*.npz is listed, and the shapeset_sha256; raises
                                     ShapesetError. Returns the manifest.
  load_shapeset(dir, verify=True, jobs=None) -> list[Shape]
                                     all jobs (or the given job ids) in job order.
  shapeset_sha256(manifest) -> str  identity of the set (sha256 over the sorted job-file sha256s).

Shape (one job; memory contract = v2/docs/FORMATS.md):
  job, name, category, n_layers        N_LAYERS to write (0 or 9 only for refuse jobs)
  desc        uint32 [nd, 16]           DESC[l][w] for slots l < nd (nd = 8 max); other slots: don't care
  expect_refuse, err_code               refuse jobs: the checker must refuse with ERR_CODE == err_code
  refuse_mode                           e.g. "QP_END" ("" for valid jobs)
  wgt  uint64 [nw]   at WGT word wgt_base + i       (layers packed from wgt_base)
  qparam uint64 [nq] at QPARAM combined word 2*qp_base + i   (word 2i = {m, q_bias}, 2i+1 = {0, s})
  act_in uint64 [ni] at ACT0 words 0..ni-1           (layer 0 input, in_sel = 0)
  out_raw     True: compare LOGIT[0..n_logits-1] with logits_expected[:n_logits] (int32)
  out_buf     ACT buffer (0/1) holding the final map when not out_raw
  out_expected uint64 [no]  expected ACT[out_buf] words 0..no-1
  out_mask    uint64 [no]   byte-enable bits per word (bit b = byte lane b is part of the map);
                            only those bytes are defined (the rest may hold anything)
  model_layer_cycles int64 [n_layers], model_total, model_mac (gos_cycle_model; model)
  shape (str), fields int64 [nd, len(field_names)], field_names, tk_total, macs
  Refuse jobs: wgt/qparam/act_in/out_* and model_layer_cycles empty; model_total = C_START + C_DONE
  (busy only for the config check), model_mac = 0.
  Shape.id is an alias of Shape.job.

Helpers: Shape.out_mask64 (bit mask per word), Shape.compare_output(words) -> (ok, n_bad_words),
Shape.compare_logits(logits16) -> bool, Shape.compare_cycles(layer_cyc, total, mac) -> bool.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT = "gos_shapeset/1"


class ShapesetError(RuntimeError):
    pass


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def shapeset_sha256(manifest: dict) -> str:
    """sha256 over the sorted (job file, sha256) list (same definition as gen_shapes.py)."""
    items = sorted((k, v["sha256"]) for k, v in manifest["files"].items() if k.startswith("jobs/"))
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def verify_manifest(d) -> dict:
    d = Path(d)
    mp = d / "shapes.json"
    if not mp.is_file():
        raise ShapesetError(f"{mp}: missing")
    man = json.loads(mp.read_text())
    if man.get("format") != FORMAT:
        raise ShapesetError(f"{mp}: format {man.get('format')!r} != {FORMAT!r}")
    files = man.get("files", {})
    listed = set(files)
    present = {f"jobs/{p.name}" for p in (d / "jobs").glob("*.npz")}
    if present - listed:
        raise ShapesetError(f"{mp}: files not in manifest: {sorted(present - listed)[:5]}")
    for rel, meta in files.items():
        p = d / rel
        if not p.is_file():
            raise ShapesetError(f"{p}: missing")
        if p.stat().st_size != meta["bytes"]:
            raise ShapesetError(f"{p}: size {p.stat().st_size} != manifest {meta['bytes']}")
        if _sha256_file(p) != meta["sha256"]:
            raise ShapesetError(f"{p}: sha256 differs from shapes.json")
    if shapeset_sha256(man) != man.get("shapeset_sha256"):
        raise ShapesetError(f"{mp}: shapeset_sha256 does not match the file list")
    if len(man.get("jobs", [])) != man.get("n_jobs"):
        raise ShapesetError(f"{mp}: n_jobs {man.get('n_jobs')} != {len(man.get('jobs', []))} job entries")
    return man


@dataclass
class Shape:
    job: int
    name: str
    category: str
    n_layers: int
    desc: np.ndarray
    expect_refuse: bool
    err_code: int
    refuse_mode: str
    wgt: np.ndarray
    wgt_base: int
    qparam: np.ndarray
    qp_base: int
    act_in: np.ndarray
    out_raw: bool
    out_buf: int
    out_expected: np.ndarray
    out_mask: np.ndarray
    logits_expected: np.ndarray
    n_logits: int
    model_layer_cycles: np.ndarray
    model_total: int
    model_mac: int
    shape: str
    fields: np.ndarray
    field_names: tuple
    tk_total: int
    macs: int
    xcheck: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> int:
        return self.job

    @property
    def qp_word_base(self) -> int:
        """First combined QPARAM word (PS view) = 2 * QP_BASE of layer 0."""
        return 2 * self.qp_base

    @property
    def out_mask64(self) -> np.ndarray:
        """uint64 bit mask per output word (0xFF in every enabled byte lane)."""
        be = self.out_mask.astype(np.uint64)
        m = np.zeros_like(be)
        for b in range(8):
            m |= ((be >> np.uint64(b)) & np.uint64(1)) * (np.uint64(0xFF) << np.uint64(8 * b))
        return m

    def layer(self, l: int) -> dict:
        return {k: int(v) for k, v in zip(self.field_names, self.fields[l])}

    def compare_output(self, words) -> tuple[bool, int]:
        """ACT[out_buf] words 0..len(out_expected)-1 read back -> (bit-exact under mask, n bad words)."""
        w = np.asarray(words, dtype=np.uint64)[:self.out_expected.size]
        if w.size != self.out_expected.size:
            return False, int(self.out_expected.size)
        bad = int(np.count_nonzero((w ^ self.out_expected) & self.out_mask64))
        return bad == 0, bad

    def compare_logits(self, logits16) -> bool:
        lg = np.asarray(logits16, dtype=np.int64)[:self.n_logits]
        return bool(np.array_equal(lg, self.logits_expected[:self.n_logits].astype(np.int64)))

    def compare_cycles(self, layer_cyc, total: int, mac: int) -> bool:
        lc = [int(x) for x in list(layer_cyc)[:self.n_layers]]
        return (lc == [int(x) for x in self.model_layer_cycles] and int(total) == self.model_total
                and int(mac) == self.model_mac)


def _scalar(z, k, cast):
    return cast(z[k][()])


def load_job(path, xcheck: bool = False) -> Shape:
    with np.load(path, allow_pickle=False) as z:
        return Shape(
            job=_scalar(z, "job", int), name=Path(path).stem, category=_scalar(z, "category", str),
            n_layers=_scalar(z, "n_layers", int), desc=z["desc"].astype(np.uint32),
            expect_refuse=bool(_scalar(z, "expect_refuse", int)), err_code=_scalar(z, "err_code", int),
            refuse_mode=_scalar(z, "refuse_mode", str),
            wgt=z["wgt"].astype(np.uint64), wgt_base=_scalar(z, "wgt_base", int),
            qparam=z["qparam"].astype(np.uint64), qp_base=_scalar(z, "qp_base", int),
            act_in=z["act_in"].astype(np.uint64), out_raw=bool(_scalar(z, "out_raw", int)),
            out_buf=_scalar(z, "out_buf", int), out_expected=z["out_expected"].astype(np.uint64),
            out_mask=z["out_mask"].astype(np.uint64), logits_expected=z["logits_expected"].astype(np.int32),
            n_logits=_scalar(z, "n_logits", int),
            model_layer_cycles=z["model_layer_cycles"].astype(np.int64),
            model_total=_scalar(z, "model_total", int), model_mac=_scalar(z, "model_mac", int),
            shape=_scalar(z, "shape", str), fields=z["fields"].astype(np.int64),
            field_names=tuple(str(x) for x in z["field_names"]), tk_total=_scalar(z, "tk_total", int),
            macs=_scalar(z, "macs", int), xcheck=xcheck)


def load_shapeset(d, verify: bool = True, jobs=None) -> list[Shape]:
    d = Path(d)
    man = verify_manifest(d) if verify else json.loads((d / "shapes.json").read_text())
    want = None if jobs is None else {int(j) for j in jobs}
    out = []
    for e in man["jobs"]:
        if want is not None and e["job"] not in want:
            continue
        s = load_job(d / e["file"], xcheck=bool(e.get("xcheck", 0)))
        if s.job != e["job"] or s.category != e["category"]:
            raise ShapesetError(f"{e['file']}: job/category differ from shapes.json")
        out.append(s)
    return out


__all__ = ["FORMAT", "Shape", "ShapesetError", "load_job", "load_shapeset", "shapeset_sha256",
           "verify_manifest"]
