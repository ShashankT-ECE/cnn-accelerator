"""A3-general board side (exp_shapes.py, s2.shapes): registration / priority, time cap, stratified
order, refuse jobs, mismatch detection, shipped-set provenance, model-backend dry run.

A small fake shape set is generated here with v2/model (gos_pack descriptors, gos_golden expected
outputs, gos_cycle_model cycles) in the real on-disk format of v2/board/shapeset.py
(gos_shapeset/1: shapes.json + jobs/J<nnnn>.npz) and loaded through the real loader. If the real
set v2/build/shapes/<seed>/ exists, a subset of it is also dry-run. Everything writes under tmp
dirs named 'dryrun' (model-output rule)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import board_common as bc
import exp_shapes as xs
import run_sessions as RS
import session_extra_steps as sx
import shapeset as ss

BOARD = Path(__file__).resolve().parents[1]
MODEL = BOARD.parent / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

import gos_cycle_model as cm  # noqa: E402
import gos_golden as gg  # noqa: E402
import gos_pack as gp  # noqa: E402

FIELD_NAMES = ("IC", "OC", "IH", "IW", "KH", "KW", "OH", "OW", "K", "OUT_H", "OUT_W", "relu_en",
               "pool_en", "out_raw", "in_sel", "WGT_BASE", "QP_BASE")

# (IC, OC, KH, KW, relu, pool, out_raw) chains on an input (IH, IW); wgt_base / qp_base offsets
CHAINS = {
    "conv": ((5, 12), [(2, 9, 3, 3, 1, 0, 0), (9, 5, 2, 2, 1, 0, 0)], (0, 0)),
    "pool": ((10, 11), [(3, 8, 3, 2, 1, 1, 0)], (100, 7)),          # 1 layer: output in ACT1
    "raw": ((6, 6), [(2, 8, 3, 3, 1, 1, 0), (8, 10, 2, 2, 0, 0, 1)], (37, 3)),
}


def _job(job, cat, in_hw, specs, bases, rng) -> ss.Shape:
    IH, IW = in_hw
    wb, qb = bases
    wb0, qb0 = wb, qb
    fields, params, wparts, qparts = [], [], [], []
    for i, (IC, OC, KH, KW, relu, pool, raw) in enumerate(specs):
        OH, OW = IH - KH + 1, IW - KW + 1
        f = {"IC": IC, "OC": OC, "IH": IH, "IW": IW, "KH": KH, "KW": KW, "OH": OH, "OW": OW,
             "WGT_BASE": wb, "QP_BASE": qb, "relu_en": relu, "pool_en": pool, "out_raw": raw,
             "in_sel": i % 2}
        f.update(gp.derive_fields(f))
        q_w = rng.integers(-20, 21, (OC, IC, KH, KW)).astype(np.int8)
        q_b = rng.integers(-500, 501, OC).astype(np.int32)
        m = rng.integers(1 << 23, 1 << 24, OC)
        s = rng.integers(26, 31, OC)
        L = {"name": f"L{i}", "IC": IC, "OC": OC, "IH": IH, "IW": IW, "KH": KH, "KW": KW,
             "OH": OH, "OW": OW, "K": f["K"], "relu": bool(relu), "pool": bool(pool),
             "final": bool(raw)}
        params.append(gg.LayerParams(L, q_w, q_b, m=tuple(int(x) for x in m),
                                     s=tuple(int(x) for x in s)))
        wparts.append(gp._pack_wgt_layer(q_w))
        qw = np.zeros(2 * OC, np.uint64)
        qw[0::2] = (m.astype(np.uint64) << np.uint64(32)) | q_b.view(np.uint32).astype(np.uint64)
        qw[1::2] = s.astype(np.uint64)
        qparts.append(qw)
        wb += f["OC_TILES"] * f["K"]
        qb += OC
        fields.append(f)
        IH, IW = f["OUT_H"], f["OUT_W"]
    desc = np.stack([gp.encode_descriptor(f) for f in fields])
    x = rng.integers(-128, 128, (specs[0][0], in_hw[0], in_hw[1])).astype(np.int8)
    y = x
    for P in params:
        y = gg.gos_layer(y, P.cfg, P)
    lc = [cm.layer_cycles(P.cfg) for P in params]
    out_raw = bool(fields[-1]["out_raw"])
    le = np.zeros(16, np.int32)
    out_exp = out_mask = np.zeros(0, np.uint64)
    if out_raw:
        le[:y.size] = y.reshape(-1)
    else:
        out_exp = gp.pack_act_words(y)
        cover = gp.pack_act(np.ones(y.shape, np.int8), gp.act_depth(*y.shape))   # uint8 [8, D]
        out_mask = (cover.astype(np.uint64) << np.arange(8, dtype=np.uint64)[:, None]).sum(0) \
            .astype(np.uint64)
    return ss.Shape(
        job=job, name=f"J{job:04d}", category=cat, n_layers=len(specs), desc=desc,
        expect_refuse=False, err_code=0, refuse_mode="", wgt=np.concatenate(wparts), wgt_base=wb0,
        qparam=np.concatenate(qparts), qp_base=qb0, act_in=gp.pack_act_words(x), out_raw=out_raw,
        out_buf=0 if out_raw else 1 - fields[-1]["in_sel"], out_expected=out_exp,
        out_mask=out_mask, logits_expected=le, n_logits=fields[-1]["OC"] if out_raw else 0,
        model_layer_cycles=np.array([c["cycles"] for c in lc], np.int64),
        model_total=sum(c["cycles"] for c in lc) + cm.C_START + cm.C_DONE,
        model_mac=sum(c["mac_active"] for c in lc), shape=cat,
        fields=np.array([[f[k] for k in FIELD_NAMES] for f in fields], np.int64),
        field_names=FIELD_NAMES, tk_total=sum(c["mac_active"] for c in lc),
        macs=sum(c["macs"] for c in lc))


def _refuse(job, base: ss.Shape, n_layers: int, desc: np.ndarray, mode: str) -> ss.Shape:
    e64, e32 = np.zeros(0, np.uint64), np.zeros(16, np.int32)
    return ss.Shape(
        job=job, name=f"J{job:04d}", category="refuse", n_layers=n_layers, desc=desc,
        expect_refuse=True, err_code=gp.job_err_code(n_layers, np.vstack(
            [desc, np.zeros((8 - desc.shape[0], 16), np.uint32)])), refuse_mode=mode,
        wgt=e64, wgt_base=0, qparam=e64, qp_base=0, act_in=e64, out_raw=False, out_buf=0,
        out_expected=e64, out_mask=e64, logits_expected=e32, n_logits=0,
        model_layer_cycles=np.zeros(0, np.int64), model_total=-1, model_mac=-1, shape="refuse",
        fields=base.fields, field_names=FIELD_NAMES, tk_total=0, macs=0)


def make_shapes(n_per_cat: int = 2, seed: int = 7, refuse: bool = True) -> list[ss.Shape]:
    rng = np.random.default_rng(seed)
    out = []
    for cat, (in_hw, specs, bases) in CHAINS.items():
        for _ in range(n_per_cat):
            out.append(_job(len(out), cat, in_hw, specs, bases, rng))
    if refuse:
        base = out[0]
        bad = base.desc.copy()
        bad[1, 7] = 4                                  # K := 4 on layer 1 -> rule 3, layer 1
        out.append(_refuse(len(out), base, 2, bad, "K_LT_8"))
        out.append(_refuse(len(out), base, 0, base.desc[:1].copy(), "N_LAYERS_0"))
        out.append(_refuse(len(out), base, 9, base.desc.copy(), "N_LAYERS_9"))
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_fake_set(d: Path, shapes, seed: int = 7) -> Path:
    """The shapeset.py on-disk format (gos_shapeset/1), read back by the real loader."""
    (d / "jobs").mkdir(parents=True, exist_ok=True)
    jobs, files = [], {}
    for s in shapes:
        rel = f"jobs/{s.name}.npz"
        np.savez(d / rel, job=s.job, category=s.category, n_layers=s.n_layers, desc=s.desc,
                 expect_refuse=int(s.expect_refuse), err_code=s.err_code,
                 refuse_mode=s.refuse_mode, wgt=s.wgt, wgt_base=s.wgt_base, qparam=s.qparam,
                 qp_base=s.qp_base, act_in=s.act_in, out_raw=int(s.out_raw), out_buf=s.out_buf,
                 out_expected=s.out_expected, out_mask=s.out_mask,
                 logits_expected=s.logits_expected, n_logits=s.n_logits,
                 model_layer_cycles=s.model_layer_cycles, model_total=s.model_total,
                 model_mac=s.model_mac, shape=s.shape, fields=s.fields,
                 field_names=np.array(s.field_names), tk_total=s.tk_total, macs=s.macs)
        files[rel] = {"sha256": _sha(d / rel), "bytes": (d / rel).stat().st_size}
        jobs.append({"job": s.job, "name": s.name, "category": s.category, "file": rel})
    man = {"format": ss.FORMAT, "seed": seed, "n_jobs": len(jobs), "git_commit": "fake",
           "git_dirty": False, "jobs": jobs, "files": files}
    man["shapeset_sha256"] = ss.shapeset_sha256(man)
    (d / "shapes.json").write_text(json.dumps(man))
    ss.verify_manifest(d)
    return d


def ship(tmp: Path, shapes, rtl_rows=None, rtl_sha=None) -> Path:
    """Fake set -> tmp/data/shapes via make_board_data.ship_shapes (+ PACKAGE.json link)."""
    import make_board_data as mbd
    src = write_fake_set(tmp / "build" / "shapes" / "7", shapes)
    (src / "hex").mkdir()
    (src / "hex" / "big.hex").write_text("00\n")                # not shipped (not listed)
    rtl = tmp / "shapes_rtl.csv"
    if rtl_rows is not None:
        sha = rtl_sha or json.loads((src / "shapes.json").read_text())["shapeset_sha256"]
        with rtl.open("w", newline="") as f:
            w = csv.DictWriter(f, ["job", "shapeset_sha256", "rtl_total", "rtl_layer_cycles",
                                   "rtl_mac", "git_dirty", "source"])
            w.writeheader()
            for r in rtl_rows:
                w.writerow({"shapeset_sha256": sha, "git_dirty": "False", "source": "rtl_sim", **r})
    data = tmp / "data"
    shp = mbd.ship_shapes(src, data / "shapes", "abc", False, rtl_csv=rtl)
    (data / "PACKAGE.json").write_text(json.dumps({
        "nets": {}, "git_dirty": False,
        "shapes": {"ship_sha256": bc.sha256_file(data / "shapes" / "SHIP.json"),
                   "shapeset_sha256": shp["shapeset_sha256"]}}))
    return data


def rtl_rows_for(shapes, bump: int | None = None):
    rows = []
    for s in shapes:
        if s.expect_refuse:
            rows.append({"job": s.job, "rtl_total": "", "rtl_layer_cycles": "", "rtl_mac": ""})
            continue
        tot = int(s.model_total) + (1 if s.job == bump else 0)
        rows.append({"job": s.job, "rtl_total": tot, "rtl_mac": int(s.model_mac),
                     "rtl_layer_cycles": ",".join(str(int(c)) for c in s.model_layer_cycles)})
    return rows


def dry_main(tmp: Path, data: Path, *extra) -> tuple[int, list[dict]]:
    out = tmp / "dryrun" / "shapes"
    rc = xs.main(["--backend", "model", "--data-dir", str(data), "--out-dir", str(out),
                  "--clock-mhz", "300", *extra])
    with (out / xs.OUT_CSV).open() as f:
        return rc, list(csv.DictReader(f))


# ---- constants -----------------------------------------------------------------------------------
def test_c_start_matches_model():
    assert xs.C_START == cm.C_START
    assert sx.SHAPES_CAP_S == xs.BUDGET_MAX_S == 900.0


# ---- stratified order ------------------------------------------------------------------------------
def test_stratified_order_covers_every_category_first():
    cats = ["a"] * 10 + ["b"] * 3 + ["c"] * 1 + ["d"] * 6
    shapes = [{"id": str(i), "category": c} for i, c in enumerate(cats)]
    o = xs.stratified_order(shapes, 5)
    assert sorted(o) == list(range(len(cats)))                      # a permutation
    assert {cats[i] for i in o[:4]} == {"a", "b", "c", "d"}          # every category first
    assert o == xs.stratified_order(shapes, 5)                        # seeded
    assert o != xs.stratified_order(shapes, 6)
    for c in "abd":                                                   # shuffled within category
        pos = [i for i in o if cats[i] == c]
        assert len(pos) == cats.count(c)


# ---- time cap (fake clock) -----------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _stub_runner(clock, dt):
    def run_one(dev, s, timeout_s=None, verify_writes=True):
        clock.t += dt
        return {"status": "error", "err_code_hw": s["err_code"], "total": xs.C_START,
                "layer_cyc": [], "mac": 0, "stall": 0, "violation": 0, "wall_ms": dt * 1e3}
    return run_one


def _refuse_dicts(n, cats=("x", "y", "z")):
    return [{"id": f"j{i}", "category": cats[i % len(cats)], "n_layers": 1,
             "desc": np.zeros((1, 16), np.uint32), "expect_refuse": True, "err_code": 0x300,
             "model_total": xs.C_START} for i in range(n)]


def test_cap_stops_cleanly_and_records_capped():
    clock = FakeClock()
    shapes = _refuse_dicts(300)
    order = xs.stratified_order(shapes, 1)
    res = xs.run_all(None, shapes, order, 900.0, {}, clock=clock, t0=0.0,
                     run_one=_stub_runner(clock, 10.0), say=lambda *_: None)
    assert res["capped"] and len(res["rows"]) == 90 and res["elapsed_s"] <= 900.0
    assert all(r["pass"] for r in res["rows"])
    assert {r["category"] for r in res["rows"][:3]} == {"x", "y", "z"}
    sm = xs.summary(res["rows"], shapes, res["capped"], res["elapsed_s"], 900.0, 1, 0)
    assert sm["jobs_completed"] == 90 and sm["jobs_total"] == 300 and sm["capped"] is True
    assert sm["categories_covered"] == 3 and sm["pass"]


def test_budget_above_hard_cap_is_capped_and_setup_time_counts():
    clock = FakeClock()
    shapes = _refuse_dicts(500)
    res = xs.run_all(None, shapes, list(range(500)), 5000.0, {}, clock=clock, t0=-100.0,
                     run_one=_stub_runner(clock, 2.0), say=lambda *_: None)
    assert res["capped"] and res["elapsed_s"] <= xs.BUDGET_MAX_S      # 100 s setup + 400 jobs
    assert len(res["rows"]) == 400


def test_uncapped_when_everything_fits():
    clock = FakeClock()
    shapes = _refuse_dicts(20)
    res = xs.run_all(None, shapes, list(range(20)), 900.0, {}, clock=clock, t0=0.0,
                     run_one=_stub_runner(clock, 1.0), say=lambda *_: None)
    assert not res["capped"] and len(res["rows"]) == 20


# ---- refuse / mismatch judgement ---------------------------------------------------------------------
def _refuse_shape(code=0x301, total=-1):
    return {"id": "r", "category": "refuse", "n_layers": 2, "desc": np.zeros((2, 16), np.uint32),
            "expect_refuse": True, "err_code": code, "model_total": total}


@pytest.mark.parametrize("r, ok", [
    ({"status": "error", "err_code_hw": 0x301, "total": 3}, True),
    ({"status": "error", "err_code_hw": 0x302, "total": 3}, False),     # wrong ERR_CODE
    ({"status": "error", "err_code_hw": 0x301, "total": 4}, False),     # TOTAL_CYC != C_START
    ({"status": "done", "err_code_hw": 0, "total": 3}, False),          # accepted
    ({"status": "timeout", "err_code_hw": 0x301, "total": 3}, False),
])
def test_refuse_job_checks(r, ok):
    row = xs.judge(_refuse_shape(), {"layer_cyc": [], **r}, None)
    assert row["refuse_ok"] is ok and row["pass"] is ok


def test_compare_act_respects_byte_mask():
    exp = np.array([0x0102030405060708, 0x1111111111111111], np.uint64)
    mask = np.array([0x0F, 0xFF], np.uint64)               # word 0: bytes 0-3 only
    got = exp.copy()
    got[0] ^= np.uint64(0xFF00000000000000)                # disabled byte 7 differs: ignored
    c = xs.compare_act(got, exp, mask)
    assert c["output_exact"] and c["bytes_checked"] == 12
    got[1] ^= np.uint64(0x0300)                            # enabled byte 1 of word 1: 0x11 -> 0x12
    c = xs.compare_act(got, exp, mask)
    assert not c["output_exact"] and c["bytes_mismatched"] == 1 and c["max_abs_err"] == 1
    with pytest.raises(ValueError):
        xs._byte_mask(np.array([0x1FF], np.uint64))


# ---- model-backend dry run on the fake set --------------------------------------------------------------
@pytest.fixture
def shapes():
    return make_shapes()


def test_model_backend_runs_arbitrary_jobs_bit_and_cycle_exact(tmp_path, shapes):
    data = ship(tmp_path, shapes, rtl_rows=rtl_rows_for(shapes))
    rc, rows = dry_main(tmp_path, data)
    jobs, sm = rows[:-1], rows[-1]
    assert rc == 0
    assert all(r["source"] == bc.SOURCE_DRYRUN for r in rows)
    assert sm["layer"] == "summary" and sm["capped"] == "False"
    assert int(sm["jobs_completed"]) == int(sm["jobs_total"]) == len(shapes)
    assert int(sm["outputs_exact"]) == len(shapes) - 3 and int(sm["refuse_ok_n"]) == 3
    assert int(sm["cycles_eq_model_n"]) == len(shapes)
    assert int(sm["cycles_eq_rtl_n"]) == int(sm["rtl_compared"]) == len(shapes) - 3
    assert sm["max_abs_err_all"] == "0" and sm["categories_covered"] == "4"
    assert {r["out_kind"] for r in jobs} == {"act0", "act1", "raw", "refuse"}
    assert all(r["pass"] == "True" for r in jobs)
    assert "DRY RUN" in sm["label"] and sm["shapeset_sha256"] and sm["ship_sha256"]


def test_output_and_cycle_mismatches_detected(tmp_path, shapes):
    act = next(s for s in shapes if not s.out_raw and not s.expect_refuse)
    en = int(np.flatnonzero(act.out_mask)[0])
    lane = int(np.flatnonzero([(int(act.out_mask[en]) >> b) & 1 for b in range(8)])[0])
    act.out_expected = act.out_expected.copy()
    act.out_expected[en] ^= np.uint64(1 << (8 * lane))      # one enabled byte off by one bit
    raw = next(s for s in shapes if s.out_raw)
    raw.model_layer_cycles = raw.model_layer_cycles + np.array([0, 1])
    rtl_bump = next(s for s in shapes if not s.expect_refuse and s is not act and s is not raw)
    data = ship(tmp_path, shapes, rtl_rows=rtl_rows_for(shapes, bump=rtl_bump.job))
    rc, rows = dry_main(tmp_path, data)
    by = {r["job_id"]: r for r in rows}
    assert rc == 1
    assert by[str(act.job)]["output_exact"] == "False" and by[str(act.job)]["bytes_mismatched"] == "1"
    assert by[str(raw.job)]["cycles_eq_model"] == "False" and by[str(raw.job)]["output_exact"] == "True"
    assert by[str(rtl_bump.job)]["cycles_eq_rtl"] == "False" and by[str(rtl_bump.job)]["pass"] == "False"
    assert by["summary"]["jobs_failed"] == "3"


def test_rtl_rows_of_another_shape_set_not_compared(tmp_path, shapes):
    data = ship(tmp_path, shapes, rtl_rows=rtl_rows_for(shapes), rtl_sha="0" * 64)
    rc, rows = dry_main(tmp_path, data)
    assert rc == 0
    assert rows[-1]["rtl_compared"] == "0" and "not compared" in rows[-1]["rtl_status"]
    assert all(r["cycles_eq_rtl"] == "" for r in rows[:-1])


def test_capped_dry_run_via_cli_budget(tmp_path, shapes):
    data = ship(tmp_path, shapes)
    rc, rows = dry_main(tmp_path, data, "--budget-s", "0")
    assert rc == 1                                      # nothing ran: not a pass
    assert rows[-1]["capped"] == "True" and rows[-1]["jobs_completed"] == "0"


# ---- provenance / shipped-set checks -----------------------------------------------------------------
def test_ship_manifest_and_package_link(tmp_path, shapes):
    data = ship(tmp_path, shapes, rtl_rows=rtl_rows_for(shapes))
    d = data / "shapes"
    s = xs.verify_ship(d)
    man = ss.verify_manifest(d)                          # the shipped copy is a valid set
    assert s["n_jobs"] == len(shapes) and s["shapeset_sha256"] == man["shapeset_sha256"]
    assert s["shapes_json_sha256"] == _sha(d / "shapes.json")
    assert set(s["files"]) == {"shapes.json", "shapes_rtl.csv", *man["files"]}
    assert not (d / "hex").exists()                     # only listed files are shipped
    assert s["rtl_csv"] == "shapes_rtl.csv" and s["categories"]["refuse"] == 3
    assert xs.check_package_link(d, s["ship_sha256"]).startswith("SHIP.json SHA256 ==")
    assert xs.main(["verify", str(d)]) == 0
    npz = next((d / "jobs").glob("*.npz"))              # tamper one job file
    npz.write_bytes(npz.read_bytes() + b"x")
    with pytest.raises(xs.ShipError, match="size|SHA256"):
        xs.verify_ship(d)
    assert xs.main(["verify", str(d)]) == 1


def test_package_link_mismatch_and_unlisted_file(tmp_path, shapes):
    data = ship(tmp_path, shapes)
    d = data / "shapes"
    pk = json.loads((data / "PACKAGE.json").read_text())
    pk["shapes"]["ship_sha256"] = "f" * 64
    (data / "PACKAGE.json").write_text(json.dumps(pk))
    with pytest.raises(xs.ShipError, match="PACKAGE.json"):
        xs.check_package_link(d, xs.verify_ship(d)["ship_sha256"])
    with pytest.raises(SystemExit, match="REFUSED"):
        dry_main(tmp_path, data)
    (d / "jobs" / "stray.npz").write_bytes(b"")
    with pytest.raises(xs.ShipError, match="not listed"):
        xs.verify_ship(d)


def test_hw_rows_refused_under_dryrun_dir(tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        bc.check_output_path(tmp_path / "dryrun" / "shapes" / xs.OUT_CSV, bc.SOURCE_HW)


# ---- session registration / priority -------------------------------------------------------------------
def _opts(data, **kw):
    return types.SimpleNamespace(**{"backend": "pynq", "data_dir": str(data), "bit": "bit/g.bit",
                                    "closed_mhz": 299.997, "results_dir": str(data.parent / "res"),
                                    "order_seed": 11, **kw})


def _ship_json(data: Path, n=300):
    (data / "shapes").mkdir(parents=True, exist_ok=True)
    (data / "shapes" / "SHIP.json").write_text(json.dumps({"n_jobs": n, "seed": "7",
                                                          "shapeset_sha256": "ab" * 32}))


def test_step_registered_only_when_shipped(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    assert "s2.shapes" not in {d["id"] for d in sx.extra_steps(_opts(data))}
    _ship_json(data)
    st = {d["id"]: d for d in sx.extra_steps(_opts(data))}["s2.shapes"]
    assert st["session"] == 2 and st["group"] == "shapes" and st["required"]
    assert st["requires"] == ["s1.smoke"] and st["outputs"] == ["hw_shapes.csv"]
    c = st["cmd"]
    assert c[0] == "exp_shapes.py" and c[c.index("--budget-s") + 1] == "900"
    assert c[c.index("--order-seed") + 1] == "41" and c[c.index("--host-path") + 1] == "safe"
    assert st["est_s"] == pytest.approx(sx.SETUP_S + 300 * sx.SHAPES_JOB_S["pynq"])
    assert 900.0 < st["timeout_s"] <= 1200.0
    assert st["params"]["shapeset_sha256"] == "ab" * 32
    fast = {d["id"]: d for d in sx.extra_steps(_opts(data, host_path="fast"))}["s2.shapes"]
    assert fast["requires"] == ["s1.smoke", "s1.fast"]
    q = {d["id"]: d for d in sx.extra_steps(_opts(data, quick=True, backend="model"))}["s2.shapes"]
    assert q["cmd"][q["cmd"].index("--budget-s") + 1] == "300" and q["est_s"] <= 300.0
    big = {d["id"]: d for d in sx.extra_steps(_opts(data, shapes_budget_s=5000))}["s2.shapes"]
    assert big["cmd"][big["cmd"].index("--budget-s") + 1] == "900"
    _ship_json(data, n=5000)                                   # job count x estimate > cap
    assert {d["id"]: d for d in sx.extra_steps(_opts(data))}["s2.shapes"]["est_s"] == 900.0
    _ship_json(data, n=20)
    small = {d["id"]: d for d in sx.extra_steps(_opts(data, backend="model"))}["s2.shapes"]
    assert small["est_s"] == pytest.approx(sx.SETUP_S + 20 * sx.SHAPES_JOB_S["model"])


def test_priority_shapes_right_after_latency(tmp_path):
    data = tmp_path / "data"
    for net in bc.NETS:
        (data / net).mkdir(parents=True)
    _ship_json(data)
    assert RS.GROUP_ORDER.index("shapes") == RS.GROUP_ORDER.index("latency") + 1
    cfg = RS.Config(backend="model", board_dir=BOARD, results_dir=tmp_path / "dryrun",
                    data_dir=data, deploy={"bit_clock_mhz": 299.997}, bit=Path("x.bit"))
    steps = RS.build_steps(cfg, [1, 2, 3])
    opts = types.SimpleNamespace(backend="model", data_dir=str(data), closed_mhz=299.997,
                                 results_dir=str(tmp_path / "dryrun"), order_seed=3)
    steps = RS.prioritize(RS.merge_extra_steps(steps, opts, [1, 2, 3], say=lambda *_: None))
    ids = [s.id for s in steps]
    k = ids.index("s2.shapes")
    assert ids.index("s2.A2A3") < ids.index("s2.A1") < ids.index("s2.B3") < k
    assert all(s.session == 1 for s in steps[:ids.index("s2.fcal")])
    after = {s.group for s in steps[k + 1:]}
    assert after <= {"baselines", "energy", "sweep", "soak", "A4", "fcal"}
    st = steps[k]
    assert st.requires == ("s1.smoke",) and st.timeout_s > 900 and st.est_s <= 900
    assert "budget_s" in st.params_key                     # resume key carries the parameters


def test_resume_key_changes_with_shape_set(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _ship_json(data)
    a = RS.step_from_dict({d["id"]: d for d in sx.extra_steps(_opts(data))}["s2.shapes"])
    (data / "shapes" / "SHIP.json").write_text(json.dumps({"n_jobs": 300, "seed": "7",
                                                          "shapeset_sha256": "cd" * 32}))
    b = RS.step_from_dict({d["id"]: d for d in sx.extra_steps(_opts(data))}["s2.shapes"])
    assert a.params_key != b.params_key
    ok, why = RS.verify_step(b, {"status": "ok", "params_key": a.params_key, "outputs": {}},
                             tmp_path)
    assert not ok and why == "parameters changed"


# ---- model backend generalization --------------------------------------------------------------------
def test_model_backend_valid_window_and_undefined_cases():
    import gos_model_backend as mb
    x = np.arange(2 * 6 * 7, dtype=np.int64).astype(np.int8).reshape(2, 6, 7)
    f = {"IH": 6, "IW": 7, "KH": 3, "KW": 2, "OH": 4, "OW": 6}          # = IH-KH+1, IW-KW+1
    assert np.array_equal(mb._valid_window(x, f), x)
    f.update(OH=2, OW=3)                                                # smaller: top-left window
    assert mb._valid_window(x, f).shape == (2, 4, 4)
    f.update(OH=5)                                                      # reads past the map
    with pytest.raises(mb.ModelUnsupported):
        mb._valid_window(x, f)


# ---- the real shape set (other owner), if generated ---------------------------------------------------
def _real_set():
    root = BOARD.parent / "build" / "shapes"
    subs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.isdigit()
                  and (d / "shapes.json").is_file()) if root.is_dir() else []
    return subs[-1] if subs else None


@pytest.mark.skipif(_real_set() is None, reason="v2/build/shapes/<seed>/ not generated")
def test_real_shape_set_dry_run(tmp_path):
    import make_board_data as mbd
    data = tmp_path / "data"
    shp = mbd.ship_shapes(_real_set(), data / "shapes", "abc", False,
                          rtl_csv=tmp_path / "absent.csv")
    (data / "PACKAGE.json").write_text(json.dumps(
        {"shapes": {"ship_sha256": bc.sha256_file(data / "shapes" / "SHIP.json")}}))
    rc, rows = dry_main(tmp_path, data)
    sm = rows[-1]
    assert rc == 0 and sm["capped"] == "False"
    assert int(sm["jobs_completed"]) == int(sm["jobs_total"]) == shp["n_jobs"]
    assert int(sm["categories_covered"]) == len(shp["categories"])
    assert int(sm["refuse_ok_n"]) == int(sm["refuse_jobs"]) == shp["categories"].get("refuse", 0)
    assert int(sm["outputs_exact"]) + int(sm["refuse_jobs"]) == shp["n_jobs"]
    assert int(sm["cycles_eq_model_n"]) == shp["n_jobs"]
