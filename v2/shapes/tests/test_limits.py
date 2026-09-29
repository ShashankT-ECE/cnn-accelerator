"""Tests for the boundary-case set (gen_limits.py + limit_cases_*.py): case loading / schema
validation, determinism, model predictions, descriptor overrides, and the collector verdict
logic with synthetic tb_shapes logs (no simulator needed).

Run: .venv/bin/python -m pytest v2/shapes/tests -q -p no:cacheprovider
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

SHAPES = Path(__file__).resolve().parents[1]
V2 = SHAPES.parent
for p in (SHAPES, V2 / "board", V2 / "model"):
    sys.path.insert(0, str(p))

import collect_shapes as cs  # noqa: E402
import gen_limits as gl  # noqa: E402
import gen_shapes as gs  # noqa: E402
import gos_pack as gp  # noqa: E402
import shapeset as ss  # noqa: E402

SEED = 777


@pytest.fixture(scope="module")
def kwset(tmp_path_factory):
    d = tmp_path_factory.mktemp("limits") / "kw"
    man = gl.generate(SEED, ["kw"], d, log=lambda *a: None)
    return d, man


def _case(**over):
    c = {"id": "T_case", "field": "KW", "value": 3, "expect": "exact", "note": "test",
         "layers": [gs.layer(2, 6, 20, 5, 3, 3)]}
    c.update(over)
    return c


# --------------------------------------------------------------------------- loading / schema
def test_loader_picks_up_every_case_module():
    mods = sorted(p.stem for p in SHAPES.glob("limit_cases_*.py"))
    assert "limit_cases_kw" in mods
    assert gl.case_modules() == mods
    cases = gl.load_cases()
    assert {c["module"] for c in cases} == set(mods)
    assert gl.case_modules(["kw"]) == ["limit_cases_kw"]
    with pytest.raises(gl.CaseError):
        gl.case_modules(["no_such_module"])


def test_case_ids_unique_and_schema_valid():
    cases = gl.load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for c in cases:
        gl.validate_case({k: v for k, v in c.items() if k != "module"})


def test_kw_module_coverage():
    cases = gl.load_cases(["kw"])
    kw = [c for c in cases if c["field"] == "KW"]
    for v in (8, 9, 10):
        assert sum(c["value"] == v for c in kw) >= 2, v
    for v in (11, 12, 16):
        assert any(c["value"] == v for c in kw), v
    assert any(c["value"] == 9 and len(c["layers"]) == 2 for c in kw)       # KW=9 in a chain
    assert any(c["value"] == 9 and c["layers"][0]["pool_en"] for c in kw)
    assert any(c["value"] == 9 and not c["layers"][0]["pool_en"] for c in kw)
    assert any(c["layers"][0]["KH"] != c["layers"][0]["KW"] for c in kw if c["value"] == 9)
    assert any(c["layers"][0]["OW"] > 16 and c["layers"][0]["OW"] % 8 for c in kw if c["value"] == 10)
    kh = {c["value"] for c in cases if c["field"] == "KH"}
    assert {9, 10, 16} <= kh


@pytest.mark.parametrize("bad", [
    {"expect": "maybe"}, {"value": "9"}, {"id": "has space"}, {"layers": []},
    {"layers": [{"IC": 1}]}, {"overrides": {"bogus": 1}}, {"overrides": {"in_sel": [0, 1]}},
    {"overrides": {"desc_patch": {0: {"NOPE": 1}}}}, {"overrides": {"desc_patch": {3: {"KW": 1}}}},
    {"overrides": {"raw_words": {0: {16: 1}}}}, {"extra_key": 1}, {"err_code": "x"},
])
def test_validate_rejects(bad):
    with pytest.raises(gl.CaseError):
        gl.validate_case(_case(**bad))


def test_missing_key_rejected():
    c = _case()
    del c["note"]
    with pytest.raises(gl.CaseError):
        gl.validate_case(c)


# --------------------------------------------------------------------------- generation
def test_determinism(kwset, tmp_path):
    d, m1 = kwset
    m2 = gl.generate(SEED, ["kw"], tmp_path / "again", log=lambda *a: None)
    assert m2["shapeset_sha256"] == m1["shapeset_sha256"] and m2["files"] == m1["files"]
    assert m2["cases"] == m1["cases"]
    for f in ("hex/jobs.hex", "hex/xcheck.hex", "hex/J0003/desc.hex", "hex/J0003/wgt.hex"):
        assert (tmp_path / "again" / f).read_bytes() == (d / f).read_bytes()
    m3 = gl.generate(SEED + 1, ["kw"], tmp_path / "other", log=lambda *a: None)
    assert m3["shapeset_sha256"] != m1["shapeset_sha256"]


def test_set_format_and_loader(kwset):
    d, man = kwset
    assert man["kind"] == gl.KIND and man["format"] == gs.FORMAT
    shapes = ss.load_shapeset(d)                       # verifies sha256s
    assert len(shapes) == man["n_jobs"] == len(man["cases"])
    assert gp.read_hex(d / "hex" / "xcheck.hex")[0] == len(shapes)
    for s, c in zip(shapes, man["cases"]):
        assert s.category == f"limit_{c['field']}"
        with np.load(d / f"jobs/{s.name}.npz") as z:
            assert str(z["case_id"]) == c["id"]


def test_model_predictions_recorded_not_asserted(kwset):
    _, man = kwset
    by = {c["id"]: c for c in man["cases"]}
    for cid, c in by.items():
        assert c["checker_accepts"] is True and c["checker_err_code"] == 0    # no KW rule (OC-3)
        kw = c["value"] if c["field"] == "KW" else None
        if kw is not None:
            # OC-3 analysis (model): the tile-model address formula is exact up to KW = 9 only
            assert c["tile_model_matches_golden"] is (kw <= 9), cid
            assert (c["tile_model_n_bad"] == 0) is (kw <= 9), cid
        assert c["model_total"] > 0 and len(c["model_layer_cycles"]) >= 1


def test_overrides_patch_and_refuse():
    base = gl.structure(_case())
    c = _case(overrides={"desc_patch": {0: {"KW": 2, "relu_en": 0}}, "raw_words": {0: {2: 0x0000_0303}}})
    st = gl.structure(c)
    assert (int(st["words"][0][2]) >> 8) & 0xFF == 3                           # raw_words after patch
    assert (int(st["words"][0][6]) >> gp.FLAG_BITS["relu_en"]) & 1 == 0
    assert np.array_equal(np.delete(st["words"][0], [2, 6]), np.delete(base["words"][0], [2, 6]))
    st = gl.structure(_case(overrides={"n_layers": 0}))
    assert st["err_code"] == gp.RULE_N_LAYERS << 8
    arr, pred = gl.build_case(SEED, 0, _case(overrides={"n_layers": 9}, expect="refuse"))
    assert int(arr["expect_refuse"]) == 1 and pred["checker_accepts"] is False
    st = gl.structure(_case(overrides={"wgt_base": 100, "qp_base": 7}))
    f = gp.decode_descriptor(st["words"][0])
    assert (f["WGT_BASE"], f["QP_BASE"]) == (100, 7)
    with pytest.raises(gl.CaseError):                    # 2 layers that do not chain
        gl.build_case(SEED, 0, _case(layers=[gs.layer(2, 6, 20, 5, 3, 3), gs.layer(3, 4, 18, 4, 3, 3)]))


# --------------------------------------------------------------------------- collector
def _line(s, **over) -> str:
    if s.expect_refuse:
        kv = dict(job=s.job, refuse=1, done=0, error=1, err_code=s.err_code, exp_code=s.err_code, err_ok=1,
                  out_ok=0, out_sig="cbf29ce484222325", n_bad=0, lg_zero=0, cycles_ok=1,
                  rtl_total=s.model_total, model_total=s.model_total, mac=0, model_mac=0, stall=0,
                  flags=0, timeout=0, layer_cycles="-", logits="-")
    else:
        sig = cs.SIG_INIT if s.out_raw else cs.out_signature(s.out_expected, s.out_mask64)
        kv = dict(job=s.job, refuse=0, done=1, error=0, err_code=0, exp_code=0, err_ok=1, out_ok=1,
                  out_sig=f"{sig:016x}", n_bad=0, lg_zero=1, cycles_ok=1, rtl_total=s.model_total,
                  model_total=s.model_total, mac=s.model_mac, model_mac=s.model_mac, stall=0, flags=0,
                  timeout=0, layer_cycles=",".join(str(int(x)) for x in s.model_layer_cycles),
                  logits=(",".join(str(int(x)) for x in s.logits_expected[:s.n_logits]) if s.out_raw else "-"))
    kv.update(over)
    return "RESULT kind=shape " + " ".join(f"{k}={v}" for k, v in kv.items())


def _make_runs(root: Path, shapes, mutate=None, drop=()):
    for sim in ("verilator", "xsim"):
        R = root / sim
        sels = [f"range:{s.job}:{s.job + 1}" for s in shapes]
        R.mkdir(parents=True, exist_ok=True)
        (R / "plan.txt").write_text("\n".join(sels) + "\n")
        for s, sel in zip(shapes, sels):
            d = R / cs.shard_dir_name(sel)
            d.mkdir(parents=True, exist_ok=True)
            if (sim, s.job) in drop:
                (d / "run.log").write_text(f"SIMULATOR name={sim} version=x\n")
                continue
            ln = _line(s)
            if mutate:
                ln = mutate(sim, s, ln)
            txt = [f"SIMULATOR name={sim} version=x", ln,
                   f"SHARD_DONE sel=range start={s.job} end={s.job + 1} jobs=1 ok=1 first_bad=-1 errors=0"]
            (d / "run.log").write_text("\n".join(txt) + "\n")
            (d / "status").write_text("0 1.0 2.0\n")
    return root


def test_classify_verdicts(kwset):
    d, _ = kwset
    s = ss.load_shapeset(d)[0]
    kv = lambda ln: cs._kv(ln)  # noqa: E731
    assert gl.classify(kv(_line(s)), s)[0] == "exact"
    assert gl.classify(kv(_line(s, n_bad=3, out_ok=0, out_sig="0" * 16)), s)[0] == "mismatch"
    assert gl.n_bad_of(kv(_line(s, n_bad=3)), s) == 3
    assert gl.classify(kv(_line(s, rtl_total=s.model_total + 1, cycles_ok=0)), s)[0] == "mismatch"
    assert gl.classify(kv(_line(s, error=1, done=1, err_code=0x301)), s)[0] == "refuse"
    assert gl.classify(kv(_line(s, timeout=1, done=0)), s)[0] == "timeout"
    assert gl.classify(None, s)[0] == "no_result"
    arr, _ = gl.build_case(SEED, 0, _case(overrides={"n_layers": 0}, expect="refuse"))
    ref = ss.Shape(**{**s.__dict__, "expect_refuse": True, "err_code": int(arr["err_code"]),
                      "model_total": int(arr["model_total"]), "model_mac": 0,
                      "model_layer_cycles": np.zeros(0, dtype=np.int64)})
    assert gl.classify(kv(_line(ref)), ref)[0] == "refuse"
    assert gl.classify(kv(_line(ref, timeout=1, error=0)), ref)[0] == "timeout"


def test_collect_end_to_end(kwset, tmp_path):
    d, man = kwset
    shapes = ss.load_shapeset(d)
    bad_job = next(c["job"] for c in man["cases"] if c["value"] == 10)

    def mism(sim, s, ln):                                   # RTL mismatch on one case (both sims)
        return _line(s, n_bad=7, out_ok=0, out_sig="0" * 16) if s.job == bad_job else ln

    R = _make_runs(tmp_path / "ok", shapes, mism)
    out = tmp_path / "ok.csv"
    assert gl.main(["collect", "--data", str(d), "--runs", str(R), "--csv", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == len(shapes)
    for r in rows:
        assert r["source"] == "rtl_sim" and r["sim_agree"] == "True"
        assert r["observed"] == ("mismatch" if r["case_id"] == man["cases"][bad_job]["id"] else "exact")
    assert rows[bad_job]["n_bad"] == "7" and rows[bad_job]["cycles_exact"] == "True"
    for k in gl.CSV_FIELDS[:15]:
        assert k in rows[0]

    def disagree(sim, s, ln):                               # the simulators differ on one case
        return _line(s, rtl_total=s.model_total + 1, cycles_ok=0) if (sim == "xsim" and s.job == 0) else ln

    R2 = _make_runs(tmp_path / "dis", shapes, disagree)
    assert gl.main(["collect", "--data", str(d), "--runs", str(R2), "--csv", str(tmp_path / "d.csv")]) == 1
    r0 = next(csv.DictReader((tmp_path / "d.csv").open()))
    assert r0["sim_agree"] == "False" and r0["observed"] == "disagree"

    R3 = _make_runs(tmp_path / "miss", shapes, drop={("verilator", 1)})   # missing RESULT
    assert gl.main(["collect", "--data", str(d), "--runs", str(R3), "--csv", str(tmp_path / "m.csv")]) == 1
    rows = list(csv.DictReader((tmp_path / "m.csv").open()))
    assert rows[1]["observed_verilator"] == "no_result"


def test_output_only_difference_on_mismatch(kwset, tmp_path):
    """Both simulators see a wrong output but different garbage-dependent values: sim_agree
    (control fields identical) but not strict, outputs_identical False. The same output
    difference on a case one simulator gets exact is a disagreement."""
    d, man = kwset
    shapes = ss.load_shapeset(d)

    def garbage_dep(sim, s, ln):
        if s.job == 0:
            return _line(s, n_bad=5 if sim == "xsim" else 4, out_ok=0,
                         out_sig=("1" if sim == "xsim" else "2") * 16)
        return ln

    R = _make_runs(tmp_path / "g", shapes, garbage_dep)
    assert gl.main(["collect", "--data", str(d), "--runs", str(R), "--csv", str(tmp_path / "g.csv")]) == 0
    r0 = next(csv.DictReader((tmp_path / "g.csv").open()))
    assert (r0["observed"], r0["sim_agree"], r0["sim_agree_strict"], r0["outputs_identical"]) == \
        ("mismatch", "True", "False", "False")
    assert r0["sim_diff_keys"] == "out_sig,n_bad" and (r0["n_bad"], r0["n_bad_xsim"]) == ("4", "5")

    def one_exact(sim, s, ln):
        return _line(s, n_bad=1, out_ok=0, out_sig="3" * 16) if (s.job == 0 and sim == "xsim") else ln

    R2 = _make_runs(tmp_path / "h", shapes, one_exact)
    assert gl.main(["collect", "--data", str(d), "--runs", str(R2), "--csv", str(tmp_path / "h.csv")]) == 1
