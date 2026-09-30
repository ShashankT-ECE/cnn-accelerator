#!/usr/bin/env python3
"""A3-general: seeded random multi-layer jobs on the accelerator (bit- and cycle-exact vs model / RTL sim).

    ./session.sh py exp_shapes.py [--shapes-dir data/shapes] [--budget-s 900] [--order-seed S]
    python3 exp_shapes.py --backend model --out-dir ../results/dryrun/shapes     # laptop dry run
    python3 exp_shapes.py verify [data/shapes]                                   # shipped-set check

Shape set: v2/shapes/gen_shapes.py (other owner) generates ~300 seeded random multi-layer jobs
into v2/build/shapes/<seed>/ (shapes.json manifest + jobs/J<nnnn>.npz; loaded with
shapeset.load_shapeset, numpy only, format gos_shapeset/1). make_board_data.py ships shapes.json +
every listed jobs/*.npz to data/shapes/ with SHIP.json (SHA256 + size of every shipped file, seed,
the set's shapeset_sha256 from shapes.json, git provenance) and, if present, a copy of the
RTL-simulation results v2/results/shapes_rtl.csv (label rtl_sim); PACKAGE.json records SHIP.json's
SHA256, so the set is covered by DEPLOY_INFO like the net packages.

Per job (driver primitives only, same register rules as every other script):
  soft_reset; WGT (from word wgt_base), QPARAM (from combined word 2*qp_base), ACT0 (= act_in,
  from word 0; host path as selected) written and (unless --no-verify-writes) read back word by
  word; DESC rows + N_LAYERS written and read back; start;
  STATUS polled until done / error (--timeout-s); LAYER_CYC[0..n-1], TOTAL_CYC, MAC_ACTIVE, STALL,
  PS_BUSY_VIOLATION, ERR_CODE read; output read back: ACT<out_buf> words 0..len(out_expected)-1
  (per-word reads) compared bit-exact on the bytes enabled by out_mask (bit b of out_mask[w] =
  byte lane b of word w; cross-checked with shapeset.Shape.compare_output), or
  LOGIT[0..n_logits-1] vs logits_expected for an out_raw job. Refuse jobs (expect_refuse; N_LAYERS
  may be 0 or 9): pass = STATUS.error, ERR_CODE == err_code and TOTAL_CYC == C_START (the shape's
  model_total when > 0; the real set stores -1). Cycles: LAYER_CYC vs
  model_layer_cycles, TOTAL_CYC vs model_total, MAC_ACTIVE vs model_mac, STALL == 0; and vs the RTL
  simulation row of the same job number in shapes_rtl.csv (rtl_layer_cycles, rtl_total, rtl_mac),
  used only if its shapeset_sha256 equals the shipped set's (RTL rows from a dirty tree are
  compared but counted in rtl_status).

Time cap: --budget-s (default and hard maximum 900 s, from script start). Before each job the
script stops if the elapsed time plus the longest job so far would exceed the budget; the summary
row then says capped=True with jobs_completed / jobs_total. Job order = a seeded shuffle within
each category, interleaved round-robin over the categories (category order shuffled too), so a
capped run still covers every category once it has run as many jobs as there are categories.

Output: hw_shapes.csv (one row per job run + one summary row, layer="summary"), board_common
CSV/provenance rules (source=hw only from the board; dry runs only under a 'dryrun' directory).
Label: "A3-general random shapes, measured on KV260" (dry run: model backend, not hardware).
Exit 0 = every job run passed (outputs, cycles vs model, and vs RTL where available); 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import board_common as bc

BUDGET_MAX_S = 900.0               # hard cap (user, 2026-09-29: 15 min)
C_START = 3                        # DECISIONS D13 (= gos_cycle_model.C_START; checked by a test)
SHIP_NAME = "SHIP.json"
SHAPES_JSON = "shapes.json"
RTL_CSV = "shapes_rtl.csv"
OUT_CSV = "hw_shapes.csv"
LABEL_HW = "A3-general random shapes, measured on KV260"
LABEL_DRY = "A3-general random shapes, DRY RUN (model backend), not hardware"
DEFAULT_ORDER_SEED = 20260929
MAX_JOB_ERRORS = 20                # exceptions / timeouts before the run aborts
# columns of v2/results/shapes_rtl.csv (other owner); first match wins
RTL_ID_COLS = ("job", "id", "job_id")
RTL_SHA_COLS = ("shapeset_sha256", "shapes_sha256", "shape_set_sha256")
RTL_TOTAL_COLS = ("rtl_total", "rtl_total_cyc", "rtl_total_cycles")
RTL_LAYER_COLS = ("rtl_layer_cycles", "rtl_layer_cyc")
RTL_MAC_COLS = ("rtl_mac", "rtl_mac_active")

FIELDS = ["label", "job_id", "job_name", "category", "order_index", "n_layers", "expect_refuse", "out_kind",
          "status", "err_code_hw", "err_code_expected", "refuse_ok", "output_exact",
          "out_words", "bytes_checked", "bytes_mismatched", "max_abs_err",
          "hw_layer_cycles", "model_layer_cycles", "rtl_layer_cycles", "hw_total", "model_total",
          "rtl_total", "hw_mac_active", "model_mac", "rtl_mac", "hw_stall", "violation", "cycles_eq_model",
          "cycles_eq_rtl", "rtl_status", "polls", "job_wall_ms", "pass", "error",
          # summary row
          "jobs_total", "jobs_completed", "capped", "budget_s", "elapsed_s", "order_seed",
          "categories_total", "categories_covered", "outputs_exact", "refuse_jobs", "refuse_ok_n",
          "cycles_eq_model_n", "cycles_eq_rtl_n", "rtl_compared", "max_abs_err_all", "jobs_failed",
          "shapeset_sha256", "ship_sha256", "shapes_seed", *bc.CLOCK_COLS]


# ---- shape access (shapeset.Shape, a dict, or any object with the fields) -------------------------
def _f(s, name, default=None):
    if isinstance(s, dict):
        return s.get(name, default)
    return getattr(s, name, default)


def job_key(s) -> str:
    """Job identity: shapeset.Shape.job (int; = the 'job' column of shapes_rtl.csv), else 'id'."""
    j = _f(s, "job")
    return str(int(j)) if j is not None else str(_f(s, "id"))


def _cycles_list(v) -> list[int]:
    if v is None:
        return []
    return [int(x) for x in np.asarray(v).reshape(-1)]


def _fmt_list(v) -> str:
    return ",".join(str(int(x)) for x in v)


# ---- shipped set -----------------------------------------------------------------------------------
class ShipError(RuntimeError):
    pass


def verify_ship(shapes_dir) -> dict:
    """Check every file listed in SHIP.json (SHA256 + size) and that no unlisted npz/json/csv is
    present. Returns SHIP.json (+ 'ship_sha256')."""
    d = Path(shapes_dir)
    p = d / SHIP_NAME
    if not p.is_file():
        raise ShipError(f"{p} missing (make_board_data.py --shapes ..., then deploy)")
    ship = json.loads(p.read_text())
    files = ship.get("files", {})
    if SHAPES_JSON not in files:
        raise ShipError(f"{p}: {SHAPES_JSON} not listed")
    for name, meta in files.items():
        f = d / name
        if not f.is_file():
            raise ShipError(f"{f} missing")
        if f.stat().st_size != meta["bytes"]:
            raise ShipError(f"{f}: size {f.stat().st_size} != SHIP.json {meta['bytes']}")
        if bc.sha256_file(f) != meta["sha256"]:
            raise ShipError(f"{f}: SHA256 differs from SHIP.json")
    extra = sorted(str(f.relative_to(d)) for f in d.rglob("*") if f.is_file()
                   and f.name != SHIP_NAME and str(f.relative_to(d)) not in files
                   and "__pycache__" not in f.parts)
    if extra:
        raise ShipError(f"{d}: files not listed in SHIP.json: {extra[:5]}")
    man = json.loads((d / SHAPES_JSON).read_text())
    if ship.get("shapeset_sha256") != man.get("shapeset_sha256"):
        raise ShipError(f"{p}: shapeset_sha256 != {SHAPES_JSON} shapeset_sha256")
    ship["ship_sha256"] = bc.sha256_file(p)
    return ship


def check_package_link(shapes_dir, ship_sha: str) -> str:
    """PACKAGE.json (parent dir) must list this SHIP.json SHA256 (chain to DEPLOY_INFO)."""
    pk = Path(shapes_dir).parent / "PACKAGE.json"
    if not pk.is_file():
        return "no PACKAGE.json beside the shape set (not chained to DEPLOY_INFO)"
    want = json.loads(pk.read_text()).get("shapes", {}).get("ship_sha256")
    if want != ship_sha:
        raise ShipError(f"SHIP.json SHA256 {ship_sha[:16]} != PACKAGE.json shapes.ship_sha256 "
                        f"{str(want)[:16]}")
    return "SHIP.json SHA256 == PACKAGE.json"


def load_shapes(shapes_dir):
    """shapeset.verify_manifest + load_shapeset (v2/board/shapeset.py, other owner)."""
    import shapeset
    shapeset.verify_manifest(shapes_dir)
    return list(shapeset.load_shapeset(shapes_dir))


# ---- RTL-simulation rows ---------------------------------------------------------------------------
def _col(row: dict, names) -> str | None:
    for n in names:
        if n in row:
            return n
    return None


def load_rtl(path, shapeset_sha: str) -> tuple[dict, str]:
    """{job id: {"total": int, "layers": [int]}} from shapes_rtl.csv rows whose shape-set SHA256
    equals shapeset_sha; plus a status string (why nothing is usable, else a count)."""
    p = Path(path)
    if not p.is_file():
        return {}, f"no {RTL_CSV} shipped"
    with p.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}, f"{RTL_CSV} empty"
    r0 = rows[0]
    cid, csha = _col(r0, RTL_ID_COLS), _col(r0, RTL_SHA_COLS)
    ctot, clay, cmac = _col(r0, RTL_TOTAL_COLS), _col(r0, RTL_LAYER_COLS), _col(r0, RTL_MAC_COLS)
    if not (cid and ctot):
        return {}, f"{RTL_CSV}: no id/total column ({sorted(r0)[:12]})"
    if not csha:
        return {}, f"{RTL_CSV}: no shape-set SHA256 column: not compared"
    out, other, dirty = {}, 0, 0
    for r in rows:
        if r[csha] != shapeset_sha:
            other += 1
            continue
        try:
            tot = int(r[ctot])
        except (TypeError, ValueError):
            continue            # a refused job (no total) or a failed sim row
        lay = [int(x) for x in r[clay].split(",") if x.strip()] if clay and r.get(clay) else []
        try:
            mac = int(r[cmac]) if cmac else None
        except (TypeError, ValueError):
            mac = None
        dirty += r.get("git_dirty", "False") in ("True", "true", "1")
        out[str(r[cid])] = {"total": tot, "layers": lay, "mac": mac}
    if not out:
        return {}, (f"{RTL_CSV}: no row for shape set {shapeset_sha[:12]} ({other} rows of other "
                    "sets): not compared")
    note = f"; {dirty} rows from a dirty tree (git_dirty=True: not paper-grade)" if dirty else ""
    return out, f"{len(out)} RTL-sim rows for shape set {shapeset_sha[:12]}{note}"


# ---- order -----------------------------------------------------------------------------------------
def stratified_order(shapes, seed: int) -> list[int]:
    """Indices into shapes: seeded shuffle within each category, categories in a seeded random
    order, then round-robin over categories (every category appears within the first
    len(categories) positions)."""
    rng = np.random.default_rng(int(seed))
    cats: dict[str, list[int]] = {}
    for i, s in enumerate(shapes):
        cats.setdefault(str(_f(s, "category", "")), []).append(i)
    names = sorted(cats)
    names = [names[k] for k in rng.permutation(len(names))]
    queues = []
    for c in names:
        idx = cats[c]
        queues.append([idx[k] for k in rng.permutation(len(idx))])
    order = []
    for k in range(max((len(q) for q in queues), default=0)):
        order += [q[k] for q in queues if k < len(q)]
    return order


# ---- one job ---------------------------------------------------------------------------------------
def _byte_mask(mask_words: np.ndarray) -> np.ndarray:
    """out_mask (byte-enable bits per 64-bit word, bit b = byte lane b) -> bool [n, 8]."""
    m = np.asarray(mask_words, dtype=np.uint64)
    if m.size and int(m.max()) > 0xFF:
        raise ValueError("out_mask holds values > 0xFF: expected 8 byte-enable bits per word")
    return ((m.reshape(-1, 1) >> np.arange(8, dtype=np.uint64)) & np.uint64(1)).astype(bool)


def compare_act(got_words, exp_words, mask_words) -> dict:
    """Bit-exact compare of int8 bytes under the byte mask; max |err| over the enabled bytes."""
    got = np.ascontiguousarray(got_words, dtype=np.uint64).view(np.int8).reshape(-1, 8)
    exp = np.ascontiguousarray(exp_words, dtype=np.uint64).view(np.int8).reshape(-1, 8)
    en = _byte_mask(mask_words)
    if not (got.shape == exp.shape == en.shape):
        raise ValueError(f"output shapes differ: got {got.shape} exp {exp.shape} mask {en.shape}")
    diff = np.abs(got.astype(np.int64) - exp.astype(np.int64))[en]
    bad = int((diff != 0).sum())
    return {"bytes_checked": int(en.sum()), "bytes_mismatched": bad,
            "max_abs_err": int(diff.max()) if diff.size else 0, "output_exact": bad == 0}


def compare_logits(got, exp, oc: int) -> dict:
    g = np.asarray(got, dtype=np.int64)[:oc]
    e = np.asarray(exp, dtype=np.int64).reshape(-1)[:oc]
    diff = np.abs(g - e)
    bad = int((diff != 0).sum())
    return {"bytes_checked": 4 * oc, "bytes_mismatched": 4 * bad,
            "max_abs_err": int(diff.max()) if diff.size else 0, "output_exact": bad == 0}


def _last_oc(desc: np.ndarray, n_layers: int) -> int:
    return (int(desc[max(0, min(n_layers, desc.shape[0])) - 1, 0]) >> 16) & 0xFFFF


def run_job(dev, s, timeout_s: float | None = None, verify_writes: bool = True) -> dict:
    """Execute one shape on the device and read everything back. Never raises for a job-level
    failure (refused / timeout / exception are recorded); the caller recovers the device."""
    import gos_driver as D
    nl = int(_f(s, "n_layers"))
    desc = np.asarray(_f(s, "desc"), dtype=np.uint32).reshape(-1, D.DESC_WORDS)
    r = {"status": "", "error": "", "polls": ""}
    t0 = time.perf_counter()
    try:
        dev.soft_reset()
        dev.write_words("WGT", np.asarray(_f(s, "wgt"), dtype=np.uint64),
                        int(_f(s, "wgt_base", 0) or 0), verify=verify_writes)
        dev.write_words("QPARAM", np.asarray(_f(s, "qparam"), dtype=np.uint64),
                        2 * int(_f(s, "qp_base", 0) or 0), verify=verify_writes)
        act = np.ascontiguousarray(_f(s, "act_in"), dtype=np.uint64)
        dev.act_in_words = 0                      # no net-length check (arbitrary input depth)
        dev.write_input(act)                      # host path as selected (safe / fast)
        if verify_writes:
            got = dev.read_words("ACT0", act.size, 0)
            bad = np.flatnonzero(got != act)
            if bad.size:
                raise D.GosError(f"ACT0 readback mismatch in {bad.size}/{act.size} words")
        dev.write_descriptors(desc[:D.MAX_LAYERS], nl, verify=True)
        try:
            polls, _ = dev.start_and_wait(timeout_s)
            r.update(status="done", polls=polls)
        except D.GosJobError as e:
            r.update(status="error", error=str(e))
        except D.GosTimeout as e:
            r.update(status="timeout", error=str(e), polls=e.polls)
        nread = max(0, min(nl, D.MAX_LAYERS))
        c = dev.read_counters(nread)
        r.update(err_code_hw=dev.err_code(), layer_cyc=[int(x) for x in c["layer_cyc"]],
                 total=int(c["total_cyc"]), mac=int(c["mac_active"]), stall=int(c["stall"]),
                 violation=int(c["violation"]))
        if r["status"] == "done" and not _f(s, "expect_refuse", False):
            if _f(s, "out_raw", False):
                r["logits"] = dev.read_logits(D.N_LOGITS).astype(np.int32)
            else:
                n = int(np.asarray(_f(s, "out_expected")).size)
                r["out_words"] = dev.read_words(f"ACT{int(_f(s, 'out_buf'))}", n, 0)
    except Exception as e:  # noqa: BLE001 - one bad job must not end the run
        r.update(status="exception", error=f"{type(e).__name__}: {e}")
    r["wall_ms"] = 1e3 * (time.perf_counter() - t0)
    return r


def judge(s, r: dict, rtl: dict | None) -> dict:
    """Row fields + pass flag for one executed job."""
    nl = int(_f(s, "n_layers"))
    refuse = bool(_f(s, "expect_refuse", False))
    desc = np.asarray(_f(s, "desc"), dtype=np.uint32).reshape(-1, 16)
    row = {"job_id": job_key(s), "job_name": str(_f(s, "name", "") or ""),
           "category": str(_f(s, "category", "")), "n_layers": nl,
           "expect_refuse": refuse, "status": r["status"], "error": r.get("error", "")[:300],
           "polls": r.get("polls", ""), "job_wall_ms": f"{r.get('wall_ms', 0.0):.3f}"}
    hw_l, hw_t = r.get("layer_cyc", []), r.get("total")
    row.update(hw_layer_cycles=_fmt_list(hw_l), hw_total="" if hw_t is None else hw_t,
               hw_mac_active=r.get("mac", ""), hw_stall=r.get("stall", ""),
               violation=r.get("violation", ""),
               err_code_hw="" if r.get("err_code_hw") is None else f"0x{r['err_code_hw']:08X}")
    if refuse:
        exp_code = int(_f(s, "err_code", 0) or 0)
        mt = _f(s, "model_total")
        exp_total = int(mt) if mt is not None and int(mt) > 0 else C_START
        ok = (r["status"] == "error" and r.get("err_code_hw") == exp_code and hw_t == exp_total)
        row.update(out_kind="refuse", err_code_expected=f"0x{exp_code:08X}", refuse_ok=ok,
                   model_total=exp_total, cycles_eq_model=hw_t == exp_total,
                   output_exact="", cycles_eq_rtl="", rtl_status="refuse job (not compared)")
        row["pass"] = ok
        return row
    ml = _cycles_list(_f(s, "model_layer_cycles"))
    mt, mm = _f(s, "model_total"), _f(s, "model_mac")
    row.update(model_layer_cycles=_fmt_list(ml), model_total="" if mt is None else int(mt),
               model_mac="" if mm is None else int(mm), err_code_expected="0x00000000")
    done = r["status"] == "done"
    if not done:
        row.update(out_kind="raw" if _f(s, "out_raw", False) else "act", output_exact=False,
                   cycles_eq_model=False, cycles_eq_rtl="", rtl_status="job not done")
        row["pass"] = False
        return row
    if _f(s, "out_raw", False):
        nlog = _f(s, "n_logits")
        nlog = int(nlog) if nlog is not None else _last_oc(desc, nl)
        cmp = compare_logits(r["logits"], _f(s, "logits_expected"), nlog)
        row.update(out_kind="raw", out_words=nlog)
        helper = getattr(s, "compare_logits", None)
        ref_ok = helper(r["logits"]) if callable(helper) else None
    else:
        cmp = compare_act(r["out_words"], _f(s, "out_expected"), _f(s, "out_mask"))
        row.update(out_kind=f"act{int(_f(s, 'out_buf'))}",
                   out_words=int(np.asarray(_f(s, "out_expected")).size))
        helper = getattr(s, "compare_output", None)
        ref_ok = helper(r["out_words"])[0] if callable(helper) else None
    if ref_ok is not None and bool(ref_ok) != bool(cmp["output_exact"]):
        cmp["output_exact"] = False            # never report exact when the two compares disagree
        row["error"] = (row["error"] + "; " if row["error"] else "") + \
            f"compare disagrees with shapeset helper ({ref_ok})"
    row.update(cmp)
    eq_model = (list(hw_l) == ml and hw_t == int(mt) and r.get("mac") == int(mm)
                and r.get("stall") == 0)
    row["cycles_eq_model"] = eq_model
    if rtl is None:
        row.update(cycles_eq_rtl="", rtl_status="no RTL-sim row")
    else:
        row.update(rtl_total=rtl["total"], rtl_layer_cycles=_fmt_list(rtl["layers"]),
                   rtl_mac="" if rtl.get("mac") is None else rtl["mac"], rtl_status="compared")
        row["cycles_eq_rtl"] = (hw_t == rtl["total"]
                                and (not rtl["layers"] or list(hw_l) == rtl["layers"])
                                and (rtl.get("mac") is None or r.get("mac") == rtl["mac"]))
    row["pass"] = (bool(cmp["output_exact"]) and eq_model and r.get("violation") == 0
                   and row["cycles_eq_rtl"] in ("", True))
    return row


# ---- the run ---------------------------------------------------------------------------------------
def run_all(dev, shapes, order, budget_s: float, rtl: dict, clock=time.monotonic, t0=None,
            run_one=run_job, timeout_s=None, verify_writes=True, recover=None,
            max_errors: int = MAX_JOB_ERRORS, say=print) -> dict:
    """Run shapes in `order` until done or the budget would be exceeded. Returns
    {"rows": [...], "capped": bool, "elapsed_s": float, "aborted": str}."""
    budget_s = min(float(budget_s), BUDGET_MAX_S)
    t0 = clock() if t0 is None else t0
    rows, longest, errors, capped, aborted = [], 0.0, 0, False, ""
    for k, i in enumerate(order):
        elapsed = clock() - t0
        if elapsed + longest > budget_s:
            capped = True
            say(f"[shapes] budget: {elapsed:.1f} s elapsed + longest job {longest:.1f} s > "
                f"{budget_s:g} s: stop after {k}/{len(order)} jobs (capped)")
            break
        s = shapes[i]
        tj = clock()
        r = run_one(dev, s, timeout_s=timeout_s, verify_writes=verify_writes)
        longest = max(longest, clock() - tj)
        row = judge(s, r, rtl.get(job_key(s)))
        row["order_index"] = k
        rows.append(row)
        if r["status"] in ("timeout", "exception") or (r["status"] == "error"
                                                        and not row["expect_refuse"]):
            errors += 1
            say(f"  [shapes] job {row['job_id']}: {r['status']} {row['error']}")
            if recover is not None:
                try:
                    recover()
                except Exception as e:  # noqa: BLE001
                    aborted = f"recover failed after job {row['job_id']}: {e}"
                    break
            if errors > max_errors:
                aborted = f"more than {max_errors} job errors"
                break
        elif not row["pass"]:
            say(f"  [shapes] job {row['job_id']} ({row['category']}): MISMATCH "
                f"output_exact={row.get('output_exact')} cycles_eq_model={row.get('cycles_eq_model')}"
                f" cycles_eq_rtl={row.get('cycles_eq_rtl')}")
        if (k + 1) % 25 == 0:
            say(f"  [shapes] {k + 1}/{len(order)} jobs, {clock() - t0:.1f} s")
    return {"rows": rows, "capped": capped, "elapsed_s": clock() - t0, "aborted": aborted}


def summary(rows: list[dict], shapes, capped: bool, elapsed: float, budget_s: float,
            seed: int, rtl_n: int) -> dict:
    ran = rows
    cats_all = {str(_f(s, "category", "")) for s in shapes}
    cats_run = {r["category"] for r in ran}
    good = [r for r in ran if not r["expect_refuse"]]
    ref = [r for r in ran if r["expect_refuse"]]
    errs = [r["max_abs_err"] for r in good if r.get("max_abs_err", "") != ""]
    return {"jobs_total": len(shapes), "jobs_completed": len(ran), "capped": capped,
            "budget_s": f"{budget_s:g}", "elapsed_s": f"{elapsed:.3f}", "order_seed": seed,
            "categories_total": len(cats_all), "categories_covered": len(cats_run),
            "outputs_exact": sum(1 for r in good if r.get("output_exact") is True),
            "refuse_jobs": len(ref), "refuse_ok_n": sum(1 for r in ref if r.get("refuse_ok")),
            "cycles_eq_model_n": sum(1 for r in ran if r.get("cycles_eq_model") is True),
            "cycles_eq_rtl_n": sum(1 for r in good if r.get("cycles_eq_rtl") is True),
            "rtl_compared": sum(1 for r in good if r.get("rtl_status") == "compared"),
            "max_abs_err_all": max(errs) if errs else "",
            "jobs_failed": sum(1 for r in ran if not r["pass"]),
            "pass": all(r["pass"] for r in ran) and bool(ran)}


class ShapesContext(bc.RunContext):
    """RunContext whose dirty flag also covers the shipped shape set."""
    shapes_dirty = False

    @property
    def dirty(self) -> bool:
        return super().dirty or bool(self.shapes_dirty)


def verify_main(argv) -> int:
    d = Path(argv[0]) if argv else bc.DEFAULT_DATA_DIR / "shapes"
    try:
        ship = verify_ship(d)
        link = check_package_link(d, ship["ship_sha256"])
        try:
            import shapeset
            shapeset.verify_manifest(d)
            man = "shapeset.verify_manifest OK"
        except ImportError:
            man = "shapeset.py not importable (manifest not checked)"
    except (ShipError, OSError, ValueError, KeyError) as e:
        print(f"shapes: FAIL {e}")
        return 1
    except Exception as e:  # noqa: BLE001 - shapeset.verify_manifest's own error types
        print(f"shapes: FAIL shapeset.verify_manifest: {type(e).__name__}: {e}")
        return 1
    print(f"shapes: OK ({len(ship['files'])} files, seed {ship.get('seed')}, n_jobs "
          f"{ship.get('n_jobs')}, set {ship['shapeset_sha256'][:12]}, commit "
          f"{str(ship.get('git_commit', ''))[:8]} dirty={ship.get('git_dirty')}; {link}; {man})")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "verify":
        return verify_main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap, nets=False)
    ap.add_argument("--shapes-dir", default=None, help="default: <data-dir>/shapes")
    ap.add_argument("--budget-s", type=float, default=BUDGET_MAX_S,
                    help=f"wall-time budget from script start (hard maximum {BUDGET_MAX_S:g} s)")
    ap.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    ap.add_argument("--no-verify-writes", action="store_true",
                    help="skip the per-word readback of WGT/QPARAM/ACT0 (DESC is always verified)")
    a = ap.parse_args(argv)
    if a.budget_s > BUDGET_MAX_S:
        print(f"[shapes] --budget-s {a.budget_s:g} > hard cap {BUDGET_MAX_S:g}: capped")
    budget = min(a.budget_s, BUDGET_MAX_S)
    t_start = time.monotonic()
    sdir = Path(a.shapes_dir) if a.shapes_dir else Path(a.data_dir) / "shapes"

    dev, be = bc.open_device(a)
    ctx = ShapesContext(a, dev, be, "A3-general")
    try:
        ship = verify_ship(sdir)
        link = check_package_link(sdir, ship["ship_sha256"])
    except ShipError as e:
        raise SystemExit(f"REFUSED: shape set: {e}") from None
    ctx.shapes_dirty = bool(ship.get("git_dirty", True))
    ctx.check_clean()
    if ctx.source == bc.SOURCE_HW and ctx.dirty and not a.allow_dirty:
        raise SystemExit("REFUSED: the shipped shape set comes from a dirty tree (SHIP.json "
                         "git_dirty) — rebuild from a clean commit, or --allow-dirty (rows invalid "
                         "for the paper)")
    ctx.banner()
    shapes = load_shapes(sdir)
    if a.limit:
        shapes = shapes[:a.limit]
    rtl, rtl_status = load_rtl(sdir / RTL_CSV, ship["shapeset_sha256"])
    order = stratified_order(shapes, a.order_seed)
    label = LABEL_HW if ctx.source == bc.SOURCE_HW else LABEL_DRY
    print(f"[shapes] {len(shapes)} jobs, {len({str(_f(s, 'category', '')) for s in shapes})} "
          f"categories; set {ship['shapeset_sha256'][:12]} (seed {ship.get('seed')}; {link}); "
          f"RTL: {rtl_status}; budget {budget:g} s; order seed {a.order_seed}")
    clk = dev.fclk0_mhz()
    res = run_all(dev, shapes, order, budget, rtl, t0=t_start, timeout_s=a.timeout_s,
                  verify_writes=not a.no_verify_writes, recover=dev.recover)
    if res["aborted"]:
        print(f"[shapes] ABORTED: {res['aborted']}", file=sys.stderr)
    ccols = ctx.clock_cols(clk)
    meta_extra = {"data_manifest_sha256": ship["ship_sha256"],
                  "data_git_commit": ship.get("git_commit", ""),
                  "data_git_dirty": ship.get("git_dirty", ""), "label": label,
                  "shapeset_sha256": ship["shapeset_sha256"], "ship_sha256": ship["ship_sha256"],
                  "shapes_seed": ship.get("seed", ""), **ccols}
    out = []
    for r in res["rows"]:
        row = ctx.meta("shapes", r["job_id"], "", 1, clock_mhz=clk)
        row.update(meta_extra)
        row.update(r)
        out.append(row)
    sm = summary(res["rows"], shapes, res["capped"], res["elapsed_s"], budget, a.order_seed, len(rtl))
    srow = ctx.meta("shapes", "summary", res["elapsed_s"], len(res["rows"]), clock_mhz=clk)
    srow.update(meta_extra)
    srow.update(sm, rtl_status=rtl_status, error=res["aborted"], job_id="summary")
    out.append(srow)
    ctx.csv(OUT_CSV, out, FIELDS)
    print(f"[shapes] {label}: {sm['jobs_completed']}/{sm['jobs_total']} jobs"
          f"{' (CAPPED at the time budget)' if sm['capped'] else ''}, categories "
          f"{sm['categories_covered']}/{sm['categories_total']}; outputs exact "
          f"{sm['outputs_exact']}/{sm['jobs_completed'] - sm['refuse_jobs']}; refuse ok "
          f"{sm['refuse_ok_n']}/{sm['refuse_jobs']}; cycles == model {sm['cycles_eq_model_n']}/"
          f"{sm['jobs_completed']}; cycles == RTL {sm['cycles_eq_rtl_n']}/{sm['rtl_compared']} "
          f"compared; max |err| {sm['max_abs_err_all']}; {sm['elapsed_s']} s")
    ok = sm["pass"] and not res["aborted"]
    print("A3-general:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
