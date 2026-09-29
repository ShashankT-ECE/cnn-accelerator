#!/usr/bin/env python3
"""Boundary-case ("limits") shape set: hand-written descriptor edge cases, same set format as
gen_shapes.py, run through the unchanged tb_shapes.sv on Verilator AND xsim (run_limits.sh).

Cases come from every module v2/shapes/limit_cases_*.py (each exposes CASES: list[dict]); the
case schema is CASE_SCHEMA below. Unlike gen_shapes.py, a case is NOT held to the generator
envelope (gen_shapes.LIMITS): it is deliberately outside it. It still goes through
gos_pack.derive_fields / encode_descriptor (host-side consistency) unless it patches the encoded
words on purpose ("desc_patch" / "raw_words"). Per case the MODEL PREDICTIONS are recorded,
never asserted: golden output (gos_golden, integer path, of the intended layers), whether the
address-level tile model (gos_tile_model, same address formulas as the RTL) reproduces the
golden, the config-checker verdict (gos_pack.job_err_code) and the model cycles
(gos_cycle_model). "expect" in a case is a prediction only; the verdict is what the RTL does.

Output (default v2/build/limits/<cases tag>/, gitignored), format gos_shapeset/1 so
v2/board/shapeset.py, tb_shapes.sv, run_shard.sh and collect_shapes.py work unchanged:
  shapes.json   manifest ("kind": "limits", "cases": per-case predictions, jobs, files, sha256s)
  jobs/J0000.npz one per case (+ case_id / case_field / case_value / case_expect arrays)
  hex/          $readmemh files for tb_shapes (hex/jobs.hex, hex/xcheck.hex = every case)
A case the checker model refuses is written as a refuse job (tb_shapes checks ERR_CODE).

Subcommands:
  gen      [--seed S] [--cases kw,fields] [--out DIR]           -> the set  (label: model)
  list     [--cases ...]                                        -> case ids
  collect  --data D --runs R [--csv PATH] [--archive PATH]      -> limits_rtl.csv (label: RTL sim)
           R holds R/verilator/ and R/xsim/ run dirs (plan.txt + shard_*/run.log, run_shard.sh).
           Exit 0 iff every case has a RESULT from both simulators and they agree (sim_agree);
           RTL mismatches are data, not failures. sim_agree_strict: every RESULT field
           (collect_shapes.XVAL_KEYS) identical. sim_agree: identical, except that the output
           VALUES (OUTPUT_KEYS) may differ when BOTH simulators observe the same "mismatch" or
           "timeout" (hang, TB soft reset, then read-back) -- a wrong output
           that depends on memory tb_shapes filled with $urandom garbage, whose sequence differs
           between Verilator and xsim; outputs_identical records whether they differ.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

SHAPES_DIR = Path(__file__).resolve().parent
V2 = SHAPES_DIR.parent
for _p in (SHAPES_DIR, V2 / "model", V2 / "board"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_shapes as gs  # noqa: E402  (reused read-only: layer(), build_job, write_hex_job, ...)
import common  # noqa: E402
import gos_fuzz  # noqa: E402
import gos_pack as gp  # noqa: E402
import gos_tile_model as tm  # noqa: E402

GENERATOR = "v2/shapes/gen_limits.py"
KIND = "limits"
DEFAULT_SEED = 20260930
DEFAULT_OUT = V2 / "build" / "limits"
CASE_GLOB = "limit_cases_*.py"
EXPECT = ("exact", "mismatch", "refuse", "unknown")
OBSERVED = ("exact", "mismatch", "refuse", "timeout", "no_result")
OUTPUT_KEYS = ("out_sig", "n_bad", "out_ok", "logits")      # output values (may read TB garbage)

# The case schema (fixed; limit_cases_*.py modules are written against it).
CASE_SCHEMA = {
    "required": {"id": str, "field": str, "value": int, "layers": list, "expect": str, "note": str},
    "optional": {"overrides": dict, "err_code": int},
    "overrides": {"wgt_base": int, "qp_base": int, "n_layers": int, "in_sel": list,
                  "desc_patch": dict, "raw_words": dict},
}
LAYER_KEYS = ("IC", "OC", "IH", "IW", "KH", "KW", "OH", "OW", "relu_en", "pool_en", "out_raw")


class CaseError(ValueError):
    """A case violates the schema or cannot be built (a bug in the case, not an RTL result)."""


# --------------------------------------------------------------------------- case loading
def case_modules(names=None) -> list[str]:
    """Module names of v2/shapes/limit_cases_*.py (sorted); `names` = suffixes to keep (kw, ...)."""
    mods = sorted(p.stem for p in SHAPES_DIR.glob(CASE_GLOB))
    if names:
        want = {f"limit_cases_{n}" if not n.startswith("limit_cases_") else n for n in names}
        missing = want - set(mods)
        if missing:
            raise CaseError(f"no case module(s) {sorted(missing)} in {SHAPES_DIR}")
        mods = [m for m in mods if m in want]
    return mods


def validate_case(c: dict) -> None:
    if not isinstance(c, dict):
        raise CaseError(f"case is not a dict: {c!r}")
    cid = c.get("id", "?")
    for k, t in CASE_SCHEMA["required"].items():
        if k not in c:
            raise CaseError(f"{cid}: missing key {k!r}")
        if not isinstance(c[k], t) or (t is int and isinstance(c[k], bool)):
            raise CaseError(f"{cid}: {k} must be {t.__name__}")
    extra = set(c) - set(CASE_SCHEMA["required"]) - set(CASE_SCHEMA["optional"])
    if extra:
        raise CaseError(f"{cid}: unknown keys {sorted(extra)}")
    if not cid or any(ch.isspace() or ch in ",;=" for ch in cid):
        raise CaseError(f"{cid!r}: id must be non-empty without whitespace , ; =")
    if c["expect"] not in EXPECT:
        raise CaseError(f"{cid}: expect {c['expect']!r} not in {EXPECT}")
    if "err_code" in c and not isinstance(c["err_code"], int):
        raise CaseError(f"{cid}: err_code must be int")
    if not 1 <= len(c["layers"]) <= gp.MAX_LAYERS:
        raise CaseError(f"{cid}: 1..{gp.MAX_LAYERS} layers")
    for i, f in enumerate(c["layers"]):
        if not isinstance(f, dict) or any(k not in f for k in LAYER_KEYS):
            raise CaseError(f"{cid}: layer {i} lacks raw fields {LAYER_KEYS}")
    ov = c.get("overrides", {})
    if not isinstance(ov, dict):
        raise CaseError(f"{cid}: overrides must be a dict")
    for k, v in ov.items():
        if k not in CASE_SCHEMA["overrides"]:
            raise CaseError(f"{cid}: unknown override {k!r}")
        if not isinstance(v, CASE_SCHEMA["overrides"][k]):
            raise CaseError(f"{cid}: override {k} must be {CASE_SCHEMA['overrides'][k].__name__}")
    n = len(c["layers"])
    if "in_sel" in ov and (len(ov["in_sel"]) != n or any(x not in (0, 1) for x in ov["in_sel"])):
        raise CaseError(f"{cid}: in_sel must list 0/1 per layer")
    for k in ("desc_patch", "raw_words"):
        for li, d in ov.get(k, {}).items():
            if not isinstance(li, int) or not 0 <= li < n or not isinstance(d, dict):
                raise CaseError(f"{cid}: {k} keys must be layer indices 0..{n - 1} -> dict")
            for fk, fv in d.items():
                if k == "desc_patch" and fk not in gp.DESC_LAYOUT and fk not in gp.FLAG_BITS:
                    raise CaseError(f"{cid}: desc_patch field {fk!r} unknown")
                if k == "raw_words" and (not isinstance(fk, int) or not 0 <= fk < gp.DESC_WORDS):
                    raise CaseError(f"{cid}: raw_words word index {fk!r}")
                if not isinstance(fv, int) or fv < 0:
                    raise CaseError(f"{cid}: {k} value must be an int >= 0")


def load_cases(names=None) -> list[dict]:
    """Every case of every selected module, in module order then list order; each case gets
    "module". Validates the schema and id uniqueness (across modules)."""
    out, seen = [], {}
    for m in case_modules(names):
        mod = importlib.import_module(m)
        cases = getattr(mod, "CASES", None)
        if not isinstance(cases, list):
            raise CaseError(f"{m}: no CASES list")
        for c in cases:
            validate_case(c)
            if c["id"] in seen:
                raise CaseError(f"duplicate case id {c['id']!r} ({seen[c['id']]}, {m})")
            seen[c["id"]] = m
            out.append({**c, "module": m})
    if not out:
        raise CaseError("no cases")
    return out


# --------------------------------------------------------------------------- building
def _patch_field(w: np.ndarray, k: str, v: int) -> None:
    if k in gp.DESC_LAYOUT:
        wi, lsb, width = gp.DESC_LAYOUT[k]
        if v >= 1 << width:
            raise CaseError(f"desc_patch {k}={v} does not fit {width} bits")
        m = ((1 << width) - 1) << lsb
        w[wi] = np.uint32((int(w[wi]) & ~m & 0xFFFFFFFF) | (v << lsb))
    else:
        bit = gp.FLAG_BITS[k]
        if v not in (0, 1):
            raise CaseError(f"desc_patch flag {k}={v}")
        w[6] = np.uint32((int(w[6]) & ~(1 << bit) & 0xFFFFFFFF) | (v << bit))


def structure(c: dict) -> dict:
    """Case -> intended layer fields (derived, bases), encoded (+patched) words, N_LAYERS, the
    checker-model ERR_CODE. No envelope rejects (gen_shapes.LIMITS is not applied)."""
    ov = c.get("overrides", {})
    Ls = [dict(f) for f in c["layers"]]
    n = len(Ls)
    for i, f in enumerate(Ls):
        f["in_sel"] = int(ov["in_sel"][i]) if "in_sel" in ov else i % 2
        f.setdefault("WGT_BASE", 0)
        f.setdefault("QP_BASE", 0)
        f.update(gp.derive_fields(f))
    acc_w, acc_q = int(ov.get("wgt_base", 0)), int(ov.get("qp_base", 0))
    for f in Ls:
        f["WGT_BASE"], f["QP_BASE"] = acc_w, acc_q
        f.update(gp.derive_fields(f))
        acc_w += f["OC_TILES"] * f["K"]
        acc_q += f["OC"]
    words = []
    for i, f in enumerate(Ls):
        try:
            words.append(gp.encode_descriptor(f))
        except AssertionError as e:
            raise CaseError(f"{c['id']}: layer {i} does not encode ({e}); use desc_patch/raw_words") from None
    words = np.stack(words).astype(np.uint32)
    for li, d in ov.get("desc_patch", {}).items():
        for k, v in d.items():
            _patch_field(words[li], k, int(v))
    for li, d in ov.get("raw_words", {}).items():
        for wi, v in d.items():
            if v >= 1 << 32:
                raise CaseError(f"{c['id']}: raw_words value {v:#x} > 32 bits")
            words[li, wi] = np.uint32(v)
    n_layers = int(ov.get("n_layers", n))
    full = np.zeros((gp.MAX_LAYERS, gp.DESC_WORDS), dtype=np.uint32)
    full[:n] = words
    err = int(gp.job_err_code(n_layers, full))
    return {"layers": Ls, "words": words, "n_layers": n_layers, "err_code": err}


def _tile_model(arr: dict, words: np.ndarray, n_run: int, rng) -> tuple[bool, str, int | str]:
    """The address-level tile model on garbage memories loaded exactly like tb_shapes (whole
    words) -> (matches the golden, note, n_bad: differing output words / logits). Never raises."""
    try:
        mem = gos_fuzz.garbage_mem(rng)
        wb, w = int(arr["wgt_base"]), arr["wgt"]
        mem.wgt[wb:wb + w.size] = w
        e, o = gp.qparam_banks(arr["qparam"])
        qb = int(arr["qp_base"])
        mem.qp_e[qb:qb + e.size] = e
        mem.qp_o[qb:qb + o.size] = o
        ai = arr["act_in"]
        mem.act[0][:, :ai.size] = gp.act_words_to_banks(ai)
        for l in range(n_run):
            tm.run_layer_mem(mem, words[l], "fast")
        if int(arr["out_raw"]):
            n = int(arr["n_logits"])
            nb = int(np.count_nonzero(mem.logit[:n] != arr["logits_expected"][:n]))
            return nb == 0, "" if nb == 0 else f"{nb}/{n} logits differ", nb
        oe = arr["out_expected"]
        got = gp.act_banks_to_words(mem.act[int(arr["out_buf"])])[:oe.size]
        be = arr["out_mask"].astype(np.uint64)
        m64 = np.zeros_like(be)
        for b in range(8):
            m64 |= ((be >> np.uint64(b)) & np.uint64(1)) * (np.uint64(0xFF) << np.uint64(8 * b))
        nb = int(np.count_nonzero((got ^ oe) & m64))
        return nb == 0, "" if nb == 0 else f"{nb}/{oe.size} words differ", nb
    except Exception as e:  # noqa: BLE001  (a model assert is a recorded prediction, not a crash)
        return False, f"tile model raised {type(e).__name__}: {str(e)[:120]}", ""


def case_rngs(seed: int, cid: str):
    k = zlib.crc32(cid.encode())
    return np.random.default_rng([seed, k, 1]), np.random.default_rng([seed, k, 3])


def build_case(seed: int, j: int, c: dict) -> tuple[dict, dict]:
    """(npz arrays, prediction record) for case j."""
    st = structure(c)
    rng, rng_tile = case_rngs(seed, c["id"])
    cat = f"limit_{c['field']}"
    n_run = st["n_layers"]
    pred = {"checker_err_code": st["err_code"], "checker_accepts": st["err_code"] == 0,
            "tile_model_matches_golden": "", "tile_model_note": "", "tile_model_n_bad": ""}
    if st["err_code"]:
        s2 = {"layers": st["layers"], "words": st["words"], "n_layers": n_run,
              "err_code": st["err_code"], "refuse": {"mode": c["id"]}}
        arr = gs.build_job(rng, j, cat, s2, False, None)
    else:
        run_L = st["layers"][:n_run]
        for a, b in zip(run_L, run_L[1:]):
            if (b["IC"], b["IH"], b["IW"]) != (a["OC"], a["OUT_H"], a["OUT_W"]):
                raise CaseError(f"{c['id']}: layers do not chain")
            if a["out_raw"]:
                raise CaseError(f"{c['id']}: out_raw layer not last")
        s2 = {"layers": run_L, "words": st["words"], "n_layers": n_run, "err_code": 0, "refuse": None}
        arr = gs.build_job(rng, j, cat, s2, False, None)
        arr.pop("_stats", None)
        ok, note, nb = _tile_model(arr, st["words"], n_run, rng_tile)
        pred.update(tile_model_matches_golden=ok, tile_model_note=note, tile_model_n_bad=nb)
    arr.pop("_stats", None)
    arr.update(case_id=np.str_(c["id"]), case_field=np.str_(c["field"]),
               case_value=np.int64(c["value"]), case_expect=np.str_(c["expect"]))
    pred.update(model_total=int(arr["model_total"]), model_mac=int(arr["model_mac"]),
                model_layer_cycles=[int(x) for x in arr["model_layer_cycles"]])
    return arr, pred


def cases_tag(names) -> str:
    return "all" if not names else "-".join(sorted(names))


def generate(seed: int = DEFAULT_SEED, names=None, out: Path | None = None, log=print) -> dict:
    t0 = time.time()
    cases = load_cases(names)
    out = Path(out) if out is not None else DEFAULT_OUT / cases_tag(names)
    for d in (out / "jobs", out / "hex"):
        if d.exists():
            shutil.rmtree(d)
    if (out / "shapes.json").exists():
        (out / "shapes.json").unlink()
    files, summary, crec = {}, [], []
    for j, c in enumerate(cases):
        a, pred = build_case(seed, j, c)
        name = f"J{j:04d}"
        p = out / "jobs" / f"{name}.npz"
        gs.save_npz_det(p, a)
        files[f"jobs/{name}.npz"] = {"sha256": gs.sha256_file(p), "bytes": p.stat().st_size}
        gs.write_hex_job(out / "hex" / name, a)
        summary.append({"job": j, "name": name, "category": str(a["category"]),
                        "n_layers": int(a["n_layers"]), "n_desc": int(a["desc"].shape[0]),
                        "expect_refuse": int(a["expect_refuse"]), "err_code": int(a["err_code"]),
                        "refuse_mode": str(a["refuse_mode"]), "shape": str(a["shape"]),
                        "out_raw": int(a["out_raw"]), "T": [int(x) for x in a["model_T"]],
                        "tk_total": int(a["tk_total"]), "model_total": int(a["model_total"]),
                        "file": f"jobs/{name}.npz", "xcheck": 1})
        crec.append({"job": j, "id": c["id"], "module": c["module"], "field": c["field"],
                     "value": c["value"], "expect": c["expect"], "err_code": c.get("err_code", ""),
                     "note": c["note"], "shape": str(a["shape"]), **pred})
    gp.write_hex(out / "hex" / "jobs.hex", [len(summary)], 32)
    gp.write_hex(out / "hex" / "xcheck.hex", [len(summary)] + list(range(len(summary))), 32)
    man = {"format": gs.FORMAT, "kind": KIND, "seed": seed, "n_jobs": len(summary),
           "case_modules": case_modules(names), "cases_tag": cases_tag(names),
           "generator": GENERATOR, "generator_sha256": gs.sha256_file(Path(__file__)),
           "case_sources_sha256": {m: gs.sha256_file(SHAPES_DIR / f"{m}.py") for m in case_modules(names)},
           "model_sources_sha256": gs.model_sources_sha256(),
           "git_commit": common.git_commit(), "git_dirty": common.git_dirty(),
           "label": "model predictions (gos_golden / gos_pack / gos_tile_model / gos_cycle_model); "
                    "not RTL, not hardware",
           "model_constants": {"C_PIPE": gs.cm.C_PIPE, "C_START": gs.cm.C_START, "C_DONE": gs.cm.C_DONE},
           "xcheck_jobs": list(range(len(summary))), "cases": crec, "files": files, "jobs": summary}
    man["shapeset_sha256"] = gs.shapeset_sha256_of(files)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shapes.json").write_text(json.dumps(man, indent=1) + "\n")
    log(f"gen_limits: {len(summary)} cases ({', '.join(man['case_modules'])}) -> {out}  "
        f"shapeset_sha256 {man['shapeset_sha256'][:16]}  ({time.time() - t0:.1f} s) [model]")
    return man


# --------------------------------------------------------------------------- collect
CSV_FIELDS = ["case_id", "field", "value", "expect", "observed", "outputs_match", "cycles_exact",
              "err_code_expected", "err_code_rtl", "tile_model_matches_golden", "checker_accepts",
              "sim_agree", "simulators", "n_bad", "note",
              "sim_agree_strict", "outputs_identical", "sim_diff_keys", "n_bad_xsim",
              "observed_verilator", "observed_xsim", "module", "shape", "n_layers", "model_total",
              "rtl_total", "model_layer_cycles", "rtl_layer_cycles", "timeout", "tile_model_n_bad",
              "tile_model_note",
              "why", "seed", "shapeset_sha256", "data_git_commit", "data_git_dirty", "run_dir",
              "logs_archive", "logs_archive_sha256"]


def classify(r: dict | None, s) -> tuple[str, dict]:
    """Observed RTL behaviour of one case from its RESULT fields: exact | mismatch | refuse |
    timeout | no_result, plus the collect_shapes.evaluate_job verdict (outputs / cycles)."""
    import collect_shapes as cs  # noqa: PLC0415
    ev = cs.evaluate_job(r, s)
    if r is None:
        return "no_result", ev
    if r.get("timeout") == "1":
        return "timeout", ev
    if r.get("error") == "1":
        return "refuse", ev
    if s.expect_refuse:                        # accepted where the checker model refuses
        return "mismatch", ev
    return ("exact" if ev["all_ok"] else "mismatch"), ev


def n_bad_of(r: dict | None, s) -> int | str:
    if r is None or s.expect_refuse:
        return ""
    if s.out_raw:
        import collect_shapes as cs  # noqa: PLC0415
        got = cs._ints(r.get("logits", ""))
        exp = [int(x) for x in s.logits_expected[:s.n_logits]]
        if len(got) != len(exp):
            return len(exp)
        return sum(a != b for a, b in zip(got, exp))
    try:
        return int(r.get("n_bad", ""))
    except ValueError:
        return ""


def sim_agreement(rv: dict | None, rx: dict | None, ov: str, ox: str) -> tuple[bool, bool, bool | str, list]:
    """(sim_agree, sim_agree_strict, outputs_identical, differing RESULT keys) of one case."""
    import collect_shapes as cs  # noqa: PLC0415
    if rv is None or rx is None:
        return False, False, "", []
    diff = [k for k in cs.XVAL_KEYS if rv.get(k) != rx.get(k)]
    strict = not diff
    agree = strict or (set(diff) <= set(OUTPUT_KEYS) and ov == ox and ov in ("mismatch", "timeout"))
    return agree, strict, not (set(diff) & set(OUTPUT_KEYS)), diff


def _fmt_bool(v):
    return "" if v == "" else bool(v)


def collect(data: Path, runs: Path, csv_path: Path, archive: Path | None = None, log=print) -> int:
    import collect_shapes as cs  # noqa: PLC0415
    import shapeset as ss  # noqa: PLC0415
    from common import base_meta, write_results_csv  # noqa: PLC0415

    man = ss.verify_manifest(data)
    if man.get("kind") != KIND:
        raise SystemExit(f"{data}: not a limits set")
    shapes = {s.job: s for s in ss.load_shapeset(data, verify=False)}
    crec = {c["job"]: c for c in man["cases"]}
    res, vers, complete = {}, {}, True
    for sim in ("verilator", "xsim"):
        R = Path(runs) / sim
        if not (R / "plan.txt").is_file():
            res[sim], vers[sim], complete = {}, "", False
            continue
        r, shards = cs._run_results(R)
        res[sim] = r
        vers[sim] = ",".join(sorted({s["version"] for s in shards if s["version"]}))
        complete &= all(s["complete"] for s in shards) and set(r) == set(shapes)
    sims = f"verilator {vers.get('verilator', '')} + xsim {vers.get('xsim', '')}"
    arch = {"logs_archive": "", "logs_archive_sha256": ""}
    if archive:
        arch = {"logs_archive": Path(archive).name, "logs_archive_sha256": cs.write_archive(Path(runs), Path(archive))}
        log(f"wrote {archive} (sha256 {arch['logs_archive_sha256']})")
    rows, all_agree = [], True
    for j in sorted(shapes):
        s, c = shapes[j], crec[j]
        rv, rx = res["verilator"].get(j), res["xsim"].get(j)
        ov, ev = classify(rv, s)
        ox, _ = classify(rx, s)
        agree, strict, same_out, diff = sim_agreement(rv, rx, ov, ox)
        all_agree &= agree
        observed = ov if (agree or ov == ox) else "disagree"
        m = base_meta(net="limits", layer=c["id"], source="rtl_sim", num_inferences=1)
        m["vivado_version"] = "2023.1"                    # xsim of Vivado 2023.1 is one of the two
        row = {"case_id": c["id"], "field": c["field"], "value": c["value"], "expect": c["expect"],
               "observed": observed,
               "outputs_match": "" if s.expect_refuse or rv is None else ev["outputs_match"],
               "cycles_exact": "" if rv is None else ev["cycles_exact"],
               "err_code_expected": c["checker_err_code"],
               "err_code_rtl": "" if rv is None else rv.get("err_code", ""),
               "tile_model_matches_golden": _fmt_bool(c["tile_model_matches_golden"]),
               "checker_accepts": c["checker_accepts"], "sim_agree": agree, "simulators": sims,
               "n_bad": n_bad_of(rv, s), "note": c["note"],
               "sim_agree_strict": strict, "outputs_identical": same_out, "sim_diff_keys": ",".join(diff),
               "n_bad_xsim": n_bad_of(rx, s),
               "observed_verilator": ov, "observed_xsim": ox, "module": c["module"],
               "shape": c["shape"], "n_layers": s.n_layers, "model_total": s.model_total,
               "rtl_total": "" if rv is None else rv.get("rtl_total", ""),
               "model_layer_cycles": ",".join(str(int(x)) for x in s.model_layer_cycles),
               "rtl_layer_cycles": "" if rv is None else rv.get("layer_cycles", ""),
               "timeout": "" if rv is None else rv.get("timeout", ""),
               "tile_model_n_bad": c["tile_model_n_bad"], "tile_model_note": c["tile_model_note"], "why": ev.get("why", ""),
               "seed": man["seed"], "shapeset_sha256": man["shapeset_sha256"],
               "data_git_commit": man["git_commit"], "data_git_dirty": man["git_dirty"],
               "run_dir": str(Path(runs).relative_to(V2)) if Path(runs).is_relative_to(V2) else str(runs),
               **arch}
        rows.append({**m, **row})
    write_results_csv(csv_path, rows, CSV_FIELDS)
    w = max(len(r["case_id"]) for r in rows)
    log(f"{'case':<{w}}  fld val  expect    verilator  xsim       tile_ok n_bad  agree strict")
    for r in rows:
        log(f"{r['case_id']:<{w}}  {r['field']:<3} {r['value']:>3}  {r['expect']:<9} "
            f"{r['observed_verilator']:<10} {r['observed_xsim']:<10} {str(r['tile_model_matches_golden']):<7} "
            f"{str(r['n_bad']):<6} {str(r['sim_agree']):<5} {r['sim_agree_strict']}")
    ok = complete and all_agree
    log(f"wrote {csv_path}  [RTL sim]")
    log(f"gen_limits collect: {len(rows)} cases, complete={complete}, simulators agree={all_agree} (strict on "
        f"{sum(r['sim_agree_strict'] is True for r in rows)}/{len(rows)}) -> "
        f"{'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- CLI
def _names(s: str | None):
    return [x for x in s.split(",") if x] if s else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED)
    g.add_argument("--cases", default=None, help="comma list of case-module suffixes (default: all)")
    g.add_argument("--out", default=None)
    li = sub.add_parser("list")
    li.add_argument("--cases", default=None)
    k = sub.add_parser("collect")
    k.add_argument("--data", required=True)
    k.add_argument("--runs", required=True)
    k.add_argument("--csv", required=True)
    k.add_argument("--archive", default="")
    a = ap.parse_args(argv)
    if a.cmd == "gen":
        m = generate(a.seed, _names(a.cases), Path(a.out) if a.out else None)
        print(json.dumps({"n_jobs": m["n_jobs"], "shapeset_sha256": m["shapeset_sha256"]}))
        return 0
    if a.cmd == "list":
        for c in load_cases(_names(a.cases)):
            print(f"{c['id']}\t{c['module']}\t{c['field']}={c['value']}\t{c['expect']}")
        return 0
    return collect(Path(a.data), Path(a.runs), Path(a.csv), Path(a.archive) if a.archive else None)


if __name__ == "__main__":
    sys.exit(main())
