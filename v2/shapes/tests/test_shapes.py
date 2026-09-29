"""Tests for the A3-general shape set: generator determinism / limits / checker validity, the
numpy-only loader (v2/board/shapeset.py), and collector mismatch detection with synthetic logs
(no simulator needed).

Run: .venv/bin/python -m pytest v2/shapes/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SHAPES = Path(__file__).resolve().parents[1]
V2 = SHAPES.parent
for p in (SHAPES, V2 / "board", V2 / "model"):
    sys.path.insert(0, str(p))

import collect_shapes as cs  # noqa: E402
import gen_shapes as gs  # noqa: E402
import gos_cycle_model as cm  # noqa: E402
import gos_golden as gg  # noqa: E402
import gos_pack as gp  # noqa: E402
import shapeset as ss  # noqa: E402

SEED = 12345
N = 40


@pytest.fixture(scope="module")
def setdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("shapes") / "set"
    gs.generate(SEED, N, d, log=lambda *a: None)
    return d


@pytest.fixture(scope="module")
def shapes(setdir):
    return ss.load_shapeset(setdir)


# --------------------------------------------------------------------------- generator
def test_determinism(setdir, tmp_path):
    m1 = json.loads((setdir / "shapes.json").read_text())
    d2 = tmp_path / "again"
    m2 = gs.generate(SEED, N, d2, log=lambda *a: None)
    assert m2["shapeset_sha256"] == m1["shapeset_sha256"]
    assert m2["files"] == m1["files"]
    for f in ("hex/xcheck.hex", "hex/J0003/meta.hex", "hex/J0003/desc.hex"):
        assert (d2 / f).read_bytes() == (setdir / f).read_bytes()
    m3 = gs.generate(SEED + 1, N, tmp_path / "other", log=lambda *a: None, hex_files=False)
    assert m3["shapeset_sha256"] != m1["shapeset_sha256"]


def test_manifest_categories_and_default_plan(setdir):
    m = json.loads((setdir / "shapes.json").read_text())
    assert m["seed"] == SEED and m["n_jobs"] == len(m["jobs"]) == sum(m["plan"].values())
    assert all(m["categories"][c] >= 1 for c in gs.CATEGORIES)
    assert sum(c for _, c in gs.PLAN) == 300 and gs.plan_counts(300) == dict(gs.PLAN)
    assert len(m["xcheck_jobs"]) >= len(gs.CATEGORIES) * gs.XCHECK_PER_CATEGORY - 1
    assert m["tile_model_checked"] is True


def test_limits_and_checker_validity(shapes):
    assert any(s.expect_refuse for s in shapes) and any(not s.expect_refuse for s in shapes)
    for s in shapes:
        if s.expect_refuse:
            full = np.zeros((8, 16), dtype=np.uint32)
            full[:len(s.desc)] = s.desc
            assert s.err_code != 0 and gp.job_err_code(s.n_layers, full) == s.err_code
            assert (s.model_total, s.model_mac) == (cm.C_START + cm.C_DONE, 0)
            assert (s.err_code >> 8) == gs.REFUSE_RULE[s.refuse_mode], s.refuse_mode
            for w in s.desc:                     # host-side field consistency still holds
                f = gp.decode_descriptor(w)
                assert np.array_equal(gp.encode_descriptor({k: f[k] for k in gp.ALL_FIELDS}), w)
            continue
        assert 1 <= s.n_layers <= 8 and s.desc.shape == (s.n_layers, 16)
        assert gp.job_err_code(s.n_layers, s.desc) == 0
        Ls = [gp.decode_descriptor(w) for w in s.desc]
        for i, (f, w) in enumerate(zip(Ls, s.desc)):
            ok, why = gp.check_descriptor(w)
            assert ok, why
            assert np.array_equal(gp.encode_descriptor({k: f[k] for k in gp.ALL_FIELDS}), w)
            assert f["in_sel"] == i % 2
            assert gp.MIN_K <= f["K"] <= gs.LIMITS["max_k"]
            assert f["KW"] <= gs.LIMITS["max_kw"] and f["KH"] <= gs.LIMITS["max_kh"]
            assert (f["OH"], f["OW"]) == (f["IH"] - f["KH"] + 1, f["IW"] - f["KW"] + 1)
            assert f["IN_END"] < gp.ACT_DEPTH
            if f["pool_en"]:
                assert f["OH"] % 2 == 0 and f["OW"] % 2 == 0
            if f["out_raw"]:
                assert i == s.n_layers - 1 and f["OC"] <= 16 and f["OH"] == f["OW"] == 1
                assert not f["relu_en"] and not f["pool_en"]
            else:
                assert f["OUT_END"] < gp.ACT_DEPTH
        for a, b in zip(Ls, Ls[1:]):
            assert (b["IC"], b["IH"], b["IW"]) == (a["OC"], a["OUT_H"], a["OUT_W"])
            assert b["WGT_BASE"] == a["WGT_END"] + 1 and b["QP_BASE"] == a["QP_END"] + 1
        assert Ls[0]["WGT_BASE"] == s.wgt_base and Ls[-1]["WGT_END"] < gp.WGT_DEPTH
        assert s.wgt_base + s.wgt.size == Ls[-1]["WGT_END"] + 1
        assert Ls[0]["QP_BASE"] == s.qp_base and Ls[-1]["QP_END"] < gp.QP_CHANNELS
        assert s.qparam.size == 2 * (Ls[-1]["QP_END"] + 1 - s.qp_base)
        assert s.act_in.size == Ls[0]["IN_END"] + 1
        assert sum(gs.tk(f) for f in Ls) == s.tk_total <= gs.LIMITS["job_tk_max"]
        # model cycles recomputed from the descriptors
        r = cm.net_cycles([{"name": f"L{i}", **{k: f[k] for k in ("IC", "OC", "OH", "OW", "K")}}
                           for i, f in enumerate(Ls)])
        assert [r["layers"][f"L{i}"]["cycles"] for i in range(len(Ls))] == list(s.model_layer_cycles)
        assert (r["total"]["cycles"], r["total"]["mac_active"]) == (s.model_total, s.model_mac)


def test_requant_params_valid(shapes):
    for s in shapes:
        if s.expect_refuse:
            continue
        for w in s.desc:
            f = gp.decode_descriptor(w)
            q_b, m, sh = gp.unpack_qparam(s.qparam, f["QP_BASE"] - s.qp_base, f["OC"])
            if f["out_raw"]:
                assert not m.any() and not sh.any()
            else:
                assert np.all((m >= np.uint64(1 << 31)) & (m < np.uint64(1 << 32)))
                assert np.all((sh >= 1) & (sh <= 63))
                assert gp.v_abs_bound(f, q_b) < 2 ** (gp.V_MUL_W - 1)


def test_packed_images_reproduce_golden(shapes):
    """Independent round trip: unpack WGT / QPARAM / ACT0 from the npz images, rerun gos_golden,
    compare with out_expected under the mask (or the logits)."""
    for s in [x for x in shapes if not x.expect_refuse][:12]:
        Ls = [gp.decode_descriptor(w) for w in s.desc]
        x = gp.unpack_act(s.act_in, Ls[0]["IC"], Ls[0]["IH"], Ls[0]["IW"])
        for i, f in enumerate(Ls):
            L = {"name": f"L{i}", **{k: f[k] for k in ("IC", "OC", "KH", "KW", "IH", "IW", "OH", "OW", "K")},
                 "relu": bool(f["relu_en"]), "pool": bool(f["pool_en"]), "final": bool(f["out_raw"])}
            w = gp.unpack_wgt(s.wgt, L, f["WGT_BASE"] - s.wgt_base)
            q_b, m, sh = gp.unpack_qparam(s.qparam, f["QP_BASE"] - s.qp_base, f["OC"])
            P = (gg.LayerParams(L, w, q_b) if f["out_raw"] else
                 gg.LayerParams(L, w, q_b, m=tuple(int(v) for v in m), s=tuple(int(v) for v in sh)))
            x = gg.gos_layer(x, L, P)
        if s.out_raw:
            assert s.compare_logits(list(x[:, 0, 0]) + [0] * (16 - s.n_logits))
        else:
            ok, bad = s.compare_output(gp.pack_act_words(x, s.out_expected.size))
            assert ok and bad == 0


def test_is_current(setdir, tmp_path):
    ok, why = gs.is_current(setdir, SEED, N)
    assert ok, why
    assert not gs.is_current(setdir, SEED + 1, N)[0]
    assert not gs.is_current(tmp_path, SEED, N)[0]


# --------------------------------------------------------------------------- loader
def test_loader_round_trip(setdir, shapes):
    m = ss.verify_manifest(setdir)
    assert len(shapes) == m["n_jobs"] and [s.job for s in shapes] == list(range(m["n_jobs"]))
    assert ss.shapeset_sha256(m) == m["shapeset_sha256"]
    sub = ss.load_shapeset(setdir, jobs=[0, 3])
    assert [s.job for s in sub] == [0, 3]
    s = next(x for x in shapes if not x.expect_refuse and not x.out_raw)
    assert s.desc.dtype == np.uint32 and s.wgt.dtype == np.uint64 and s.out_mask.dtype == np.uint64
    assert s.out_mask.max() <= 255 and s.out_mask.size == s.out_expected.size
    m64 = s.out_mask64
    for b in range(8):
        on = (s.out_mask >> np.uint64(b)) & np.uint64(1)
        assert np.array_equal((m64 >> np.uint64(8 * b)) & np.uint64(0xFF), on * np.uint64(0xFF))
    ok, bad = s.compare_output(s.out_expected)
    assert ok and bad == 0
    i = int(np.flatnonzero(s.out_mask)[0])
    b = int(np.flatnonzero([(int(s.out_mask[i]) >> k) & 1 for k in range(8)])[0])
    w = s.out_expected.copy()
    w[i] ^= np.uint64(1 << (8 * b))
    assert s.compare_output(w) == (False, 1)
    un = np.flatnonzero(s.out_mask != 255)
    if un.size:                                   # a byte outside the map is don't-care
        j = int(un[0])
        k = next(k for k in range(8) if not (int(s.out_mask[j]) >> k) & 1)
        w = s.out_expected.copy()
        w[j] ^= np.uint64(0xFF << (8 * k))
        assert s.compare_output(w)[0]
    assert s.compare_cycles(s.model_layer_cycles, s.model_total, s.model_mac)
    assert not s.compare_cycles(s.model_layer_cycles, s.model_total + 1, s.model_mac)
    assert s.layer(0)["K"] == s.fields[0][list(s.field_names).index("K")]
    r = next(x for x in shapes if x.out_raw)
    assert r.compare_logits(r.logits_expected) and 1 <= r.n_logits <= 16


def test_loader_has_no_model_or_torch_imports():
    import ast
    tree = ast.parse((V2 / "board" / "shapeset.py").read_text())
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add((n.module or "").split(".")[0])
    assert mods <= {"__future__", "hashlib", "json", "dataclasses", "pathlib", "numpy"}, mods


def test_verify_manifest_detects_tampering(setdir, tmp_path):
    d = tmp_path / "copy"
    shutil.copytree(setdir, d, ignore=shutil.ignore_patterns("hex"))
    ss.verify_manifest(d)
    p = d / "jobs" / "J0001.npz"
    b = bytearray(p.read_bytes())
    b[-10] ^= 1
    p.write_bytes(bytes(b))
    with pytest.raises(ss.ShapesetError):
        ss.verify_manifest(d)
    shutil.copy(setdir / "jobs" / "J0001.npz", p)
    ss.verify_manifest(d)
    shutil.copy(p, d / "jobs" / "J9999.npz")
    with pytest.raises(ss.ShapesetError, match="not in manifest"):
        ss.verify_manifest(d)
    (d / "jobs" / "J9999.npz").unlink()
    m = json.loads((d / "shapes.json").read_text())
    m["shapeset_sha256"] = "0" * 64
    (d / "shapes.json").write_text(json.dumps(m))
    with pytest.raises(ss.ShapesetError, match="shapeset_sha256"):
        ss.verify_manifest(d)


# --------------------------------------------------------------------------- collector
def result_line(s, **over) -> str:
    """A passing RESULT line for Shape s (as tb_shapes prints it), with overrides."""
    if s.expect_refuse:
        kv = dict(job=s.job, refuse=1, done=0, error=1, err_code=s.err_code, exp_code=s.err_code,
                  err_ok=1, out_ok=0, out_sig=f"{cs.SIG_INIT:016x}", n_bad=0, lg_zero=0, cycles_ok=1,
                  rtl_total=s.model_total, model_total=s.model_total, mac=0, model_mac=0, stall=0,
                  flags=0, timeout=0,
                  layer_cycles="-", logits="-")
    else:
        sig = cs.SIG_INIT if s.out_raw else cs.out_signature(s.out_expected, s.out_mask64)
        kv = dict(job=s.job, refuse=0, done=1, error=0, err_code=0, exp_code=0, err_ok=1, out_ok=1,
                  out_sig=f"{sig:016x}", n_bad=0, lg_zero=1, cycles_ok=1, rtl_total=s.model_total,
                  model_total=s.model_total, mac=s.model_mac, model_mac=s.model_mac, stall=0, flags=0,
                  timeout=0, layer_cycles=",".join(str(int(x)) for x in s.model_layer_cycles),
                  logits=(",".join(str(int(x)) for x in s.logits_expected[:s.n_logits]) if s.out_raw else "-"))
    kv.update(over)
    return "RESULT kind=shape " + " ".join(f"{k}={v}" for k, v in kv.items())


def write_shard(d: Path, sel: str, lines: list[str], jobs: int, sim="verilator", done=True):
    d.mkdir(parents=True, exist_ok=True)
    txt = [f"SIMULATOR name={sim} version=5.028", "SHAPES_START"] + lines
    if done:
        txt += [f"SHARD_DONE sel=range start=0 end={jobs} jobs={jobs} ok={jobs} first_bad=-1 errors=0",
                f"TEST PASSED jobs={jobs}"]
    (d / "run.log").write_text("\n".join(txt) + "\n")
    (d / "status").write_text("0 100.0 101.5\n")


def make_run(root: Path, shapes, sels, mutate=None, sim="verilator") -> Path:
    R = root / sim
    R.mkdir(parents=True, exist_ok=True)
    (R / "plan.txt").write_text("\n".join(sels) + "\n")
    by = {s.job: s for s in shapes}
    for sel in sels:
        js = cs.sel_jobs(sel)
        lines = []
        for j in js:
            ln = result_line(by[j])
            if mutate:
                ln = mutate(j, ln, by[j])
            if ln is not None:
                lines.append(ln)
        write_shard(R / cs.shard_dir_name(sel), sel, lines, len(js), sim=sim)
    return R


def test_evaluate_all_ok(shapes):
    for s in shapes:
        ev = cs.evaluate_job(cs._kv(result_line(s)), s)
        assert ev["all_ok"], (s.job, ev["why"])


@pytest.mark.parametrize("what", ["sig", "n_bad", "tb_out", "logits", "lg_zero", "cycles", "layer",
                                  "mac", "stall", "flags", "timeout", "error", "refuse_code",
                                  "refuse_busy", "refuse_cycles", "refuse_mac"])
def test_evaluate_detects_mismatch(shapes, what):
    act = next(s for s in shapes if not s.expect_refuse and not s.out_raw)
    raw = next(s for s in shapes if s.out_raw)
    ref = next(s for s in shapes if s.expect_refuse)
    s, over = {
        "sig": (act, {"out_sig": "0123456789abcdef"}),
        "n_bad": (act, {"n_bad": 1}),
        "tb_out": (act, {"out_ok": 0}),
        "logits": (raw, {"logits": ",".join(str(int(x) + (i == 0)) for i, x in
                                            enumerate(raw.logits_expected[:raw.n_logits]))}),
        "lg_zero": (raw, {"lg_zero": 0}),
        "cycles": (act, {"rtl_total": act.model_total + 1}),
        "layer": (act, {"layer_cycles": ",".join(str(int(x) + 1) for x in act.model_layer_cycles)}),
        "mac": (act, {"mac": act.model_mac - 1}),
        "stall": (act, {"stall": 2}),
        "flags": (act, {"flags": 16}),
        "timeout": (act, {"timeout": 1}),
        "error": (act, {"error": 1, "err_code": 256}),
        "refuse_code": (ref, {"err_code": ref.err_code + 1}),
        "refuse_busy": (ref, {"timeout": 1}),
        "refuse_cycles": (ref, {"rtl_total": ref.model_total + 1}),
        "refuse_mac": (ref, {"mac": 5}),
    }[what]
    ev = cs.evaluate_job(cs._kv(result_line(s, **over)), s)
    assert not ev["all_ok"], what
    assert cs.evaluate_job(None, s)["all_ok"] is False


def test_shard_completeness(shapes, tmp_path):
    by = {s.job: s for s in shapes}
    sel = "range:0:5"
    lines = [result_line(by[j]) for j in range(5)]
    write_shard(tmp_path / "a", sel, lines, 5)
    assert cs.load_shard(tmp_path / "a", sel)["complete"]
    write_shard(tmp_path / "b", sel, lines[:4], 5)                     # missing job
    assert not cs.load_shard(tmp_path / "b", sel)["complete"]
    write_shard(tmp_path / "c", sel, lines + [lines[0]], 5)            # duplicate job
    assert "duplicate" in cs.load_shard(tmp_path / "c", sel)["why"]
    write_shard(tmp_path / "d", sel, lines, 5, done=False)             # truncated log
    assert not cs.load_shard(tmp_path / "d", sel)["complete"]
    lst = tmp_path / "l.hex"
    lst.write_text("00000002\n00000003\n00000001\n")
    assert cs.sel_jobs(f"list:{lst}") == [3, 1]
    write_shard(tmp_path / "e", f"list:{lst}", [result_line(by[3]), result_line(by[1])], 2)
    assert cs.load_shard(tmp_path / "e", f"list:{lst}")["complete"]
    assert cs.main(["check-shard", str(tmp_path / "a"), sel]) == 0
    assert cs.main(["check-shard", str(tmp_path / "b"), sel]) == 1


def test_collect_end_to_end_and_mismatch(setdir, shapes, tmp_path):
    sels = [f"range:{a}:{min(a + 16, N)}" for a in range(0, N, 16)]
    R = make_run(tmp_path / "ok", shapes, sels)
    out = tmp_path / "ok.csv"
    assert cs.main(["collect", "--run-dir", str(R), "--data", str(setdir), "--csv", str(out),
                    "--shard-csv", str(tmp_path / "sh.csv"), "--archive", str(tmp_path / "logs.tar.xz")]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == N and all(r["all_ok"] == "True" and r["source"] == "rtl_sim" for r in rows)
    assert {r["net"] for r in rows} == {"shapes"} and rows[0]["layer"] == "J0000"
    assert rows[0]["logs_archive_sha256"] and rows[0]["shapeset_sha256"]
    valid = [r for r in rows if r["expect_refuse"] == "0"]
    assert all(r["total_err"] == "0" and r["rtl_total"] == r["model_total"] for r in valid)
    # one wrong output signature, one missing job -> exit 1, rows flagged
    bad_job = next(s.job for s in shapes if not s.expect_refuse and not s.out_raw)

    missing = next(s.job for s in shapes if s.job != bad_job)

    def mut(j, ln, s):
        if j == bad_job:
            return re.sub(r"out_sig=\w+", "out_sig=0000000000000001", ln)
        return None if j == missing else ln
    R2 = make_run(tmp_path / "bad", shapes, sels, mutate=mut)
    out2 = tmp_path / "bad.csv"
    assert cs.main(["collect", "--run-dir", str(R2), "--data", str(setdir), "--csv", str(out2)]) == 1
    rows2 = {int(r["job_id"]): r for r in csv.DictReader(open(out2))}
    assert rows2[bad_job]["all_ok"] == "False" and rows2[bad_job]["outputs_match"] == "False"
    assert rows2[missing]["all_ok"] == "False"
    assert sum(r["all_ok"] == "True" for r in rows2.values()) == N - 2


def test_xval_compare(setdir, shapes, tmp_path):
    xs = json.loads((setdir / "shapes.json").read_text())["xcheck_jobs"]
    lst = tmp_path / "x0.hex"
    lst.write_text("".join(f"{v:08x}\n" for v in [len(xs)] + xs))
    sel = [f"list:{lst}"]
    A = make_run(tmp_path / "x", shapes, sel, sim="xsim")
    B = make_run(tmp_path / "x", shapes, sel, sim="verilator")
    j = tmp_path / "xv.json"
    assert cs.main(["xval", "--a", str(A), "--b", str(B), "--data", str(setdir), "--min-jobs", "5",
                    "--out", str(j)]) == 0
    assert json.loads(j.read_text())["ok"]
    assert cs.main(["xval", "--a", str(A), "--b", str(B), "--data", str(setdir),
                    "--min-jobs", str(len(xs) + 1)]) == 1
    tgt = next(x for x in xs if not shapes[x].expect_refuse)
    C = make_run(tmp_path / "y", shapes, sel, sim="verilator",
                 mutate=lambda jj, ln, s: ln.replace(f"mac={s.model_mac} ", f"mac={s.model_mac + 1} ")
                 if jj == tgt else ln)
    assert cs.main(["xval", "--a", str(A), "--b", str(C), "--data", str(setdir)]) == 1
    # collect marks the cross-checked jobs
    R = make_run(tmp_path / "z", shapes, [f"range:0:{N}"])
    out = tmp_path / "z.csv"
    assert cs.main(["collect", "--run-dir", str(R), "--data", str(setdir), "--csv", str(out),
                    "--xval-json", str(j)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert {int(r["job_id"]) for r in rows if r["xsim_crosschecked"] == "1"} == set(xs)
