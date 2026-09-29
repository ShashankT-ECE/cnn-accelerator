"""Tests for v2/fullsim/collect_full10k.py with synthetic shard logs (no simulator needed).

Run: .venv/bin/python -m pytest v2/fullsim/tests -q
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

FULLSIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FULLSIM))
import collect_full10k as cf  # noqa: E402

OC = 10
NL = 5
TOTAL = 16436


def gold(n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    v = rng.integers(-5000, 5000, size=(n, OC)).astype(np.int32)
    pred = v.argmax(1)
    labels = pred.copy()
    labels[::3] = (labels[::3] + 1) % OC           # some wrong golden predictions
    return {"v": v, "pred": pred, "labels": labels}


def argmax_predict(v):
    return np.asarray(v).argmax(1)


def result_line(net, img, logits, logits_ok=1, cycles_ok=1, total=TOTAL):
    lg = ",".join(str(int(x)) for x in list(logits) + [0] * (16 - len(logits)))
    return (f"RESULT kind=full net={net} img={img} logits_ok={logits_ok} cycles_ok={cycles_ok} "
            f"rtl_total={total} model_total={TOTAL} mac=16288 stall=0 logits={lg} "
            f"layer_cycles=2829,6029,6029,1349,197")


def make_log(net, start, end, g, *, bad_logit=(), tb_bad=(), bad_cyc=(), skip=(), dup=(),
             done=True, done_range=None, sim="verilator"):
    lines = [f"SIMULATOR name={sim} version=5.028", f"FULL10K_START net={net} start={start} end={end}"]
    errors = first = 0
    first = -1
    for i in range(start, end):
        if i in skip:
            continue
        lg = g["v"][i].copy()
        if i in bad_logit:
            lg[3] += 1                              # RTL differs, TB (wrongly) claims ok
        lok = 0 if i in tb_bad else 1
        cok = 0 if i in bad_cyc else 1
        lines.append(result_line(net, i, lg, lok, cok))
        if i in dup:
            lines.append(result_line(net, i, lg, lok, cok))
        if not (lok and cok):
            errors += 1
            first = i if first < 0 else first
    if done:
        s, e = done_range or (start, end)
        lines.append(f"SHARD_DONE net={net} start={s} end={e} images={e - s} logits_ok=0 "
                     f"cycles_ok=0 first_bad={first} errors={errors}")
        lines.append("TEST PASSED images=0 net=x" if errors == 0 else "TEST FAILED net=x errors=1")
    return "\n".join(lines) + "\n"


def write_shard(root: Path, net, start, end, text, status=(0, 100.0, 110.0)):
    d = root / net / cf.shard_dir_name(start, end)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.log").write_text(text)
    if status is not None:
        (d / "status").write_text(" ".join(str(x) for x in status) + "\n")
    return d


def write_plan(root: Path, net, plan):
    (root / net).mkdir(parents=True, exist_ok=True)
    (root / net / "plan.txt").write_text("".join(f"{s} {e}\n" for s, e in plan))


# --------------------------------------------------------------------------- completeness
def test_complete_shard():
    g = gold(20)
    p = cf.parse_log(make_log("lenet5", 0, 10, g))
    assert p["sim"] == "verilator" and p["version"] == "5.028" and p["passed"]
    assert cf.shard_complete(p, "lenet5", 0, 10) == (True, "complete")


@pytest.mark.parametrize("kw,why", [
    ({"done": False}, "no SHARD_DONE"),
    ({"done_range": (0, 9)}, "SHARD_DONE mismatch"),
    ({"skip": (4,)}, "RESULT images != range"),
    ({"dup": (2,)}, "duplicate"),
])
def test_incomplete_shard(kw, why):
    g = gold(20)
    p = cf.parse_log(make_log("lenet5", 0, 10, g, **kw))
    ok, reason = cf.shard_complete(p, "lenet5", 0, 10)
    assert not ok and why in reason


def test_wrong_net_or_range_is_incomplete():
    g = gold(20)
    p = cf.parse_log(make_log("lenet5", 0, 10, g))
    assert not cf.shard_complete(p, "cifar10", 0, 10)[0]
    assert not cf.shard_complete(p, "lenet5", 10, 20)[0]


def test_truncated_log_is_incomplete():
    g = gold(20)
    text = make_log("lenet5", 0, 10, g)
    cut = text[:text.index("RESULT kind=full net=lenet5 img=6")]
    assert not cf.shard_complete(cf.parse_log(cut), "lenet5", 0, 10)[0]


def test_check_shard_cli_resume(tmp_path):
    """run_full10k.sh skips a shard iff check-shard exits 0."""
    g = gold(20)
    d_ok = write_shard(tmp_path, "lenet5", 0, 10, make_log("lenet5", 0, 10, g))
    d_bad = write_shard(tmp_path, "lenet5", 10, 20, make_log("lenet5", 10, 20, g, done=False))
    d_fail = write_shard(tmp_path, "cifar10", 0, 10, make_log("cifar10", 0, 10, g, tb_bad=(3,)))
    assert cf.main(["check-shard", str(d_ok), "lenet5", "0", "10"]) == 0
    assert cf.main(["check-shard", str(d_bad), "lenet5", "10", "20"]) == 1
    assert cf.main(["check-shard", str(tmp_path / "missing"), "lenet5", "0", "10"]) == 1
    # complete but failing: kept (deterministic result), not re-run; the collector reports it
    assert cf.main(["check-shard", str(d_fail), "cifar10", "0", "10"]) == 0


# --------------------------------------------------------------------------- evaluation
def shards_for(root, net, plan, g, **kw_by_start):
    out = []
    for s, e in plan:
        write_shard(root, net, s, e, make_log(net, s, e, g, **kw_by_start.get(str(s), {})),
                    status=(0, 100.0 + s, 110.0 + s))
        out.append(cf.load_shard(root / net / cf.shard_dir_name(s, e), net, s, e))
    return out


def test_all_ok(tmp_path):
    g = gold(20)
    sh = shards_for(tmp_path, "lenet5", [(0, 10), (10, 20)], g)
    r = cf.evaluate_net("lenet5", sh, g, argmax_predict)
    assert r["all_ok"] and r["images"] == 20 and r["logits_bitexact"] == 20
    assert r["pred_match"] == 20 and r["cycles_exact"] == 20 and r["first_mismatch_img"] == ""
    assert r["shards_complete"] == 2 and r["simulator"] == "verilator"
    assert r["rtl_correct"] == r["golden_correct"] == int((g["pred"] == g["labels"]).sum())
    assert r["wall_s"] == 20.0 and r["shard_seconds_sum"] == 20.0


def test_logit_mismatch_detected_even_if_tb_says_ok(tmp_path):
    g = gold(20)
    sh = shards_for(tmp_path, "lenet5", [(0, 10), (10, 20)], g, **{"10": {"bad_logit": (15, 12)}})
    r = cf.evaluate_net("lenet5", sh, g, argmax_predict)
    assert not r["all_ok"] and r["logits_bitexact"] == 18 and r["first_mismatch_img"] == 12
    assert r["mismatch_imgs"] == [12, 15]


def test_tb_mismatch_and_cycles(tmp_path):
    g = gold(20)
    sh = shards_for(tmp_path, "lenet5", [(0, 10), (10, 20)], g,
                    **{"0": {"bad_cyc": (7,)}, "10": {"tb_bad": (11,)}})
    r = cf.evaluate_net("lenet5", sh, g, argmax_predict)
    assert not r["all_ok"] and r["cycles_exact"] == 19 and r["logits_bitexact"] == 19
    assert r["first_mismatch_img"] == 7 and r["n_mismatch"] == 2


def test_prediction_mismatch(tmp_path):
    g = gold(10)
    sh = shards_for(tmp_path, "lenet5", [(0, 10)], g)
    r = cf.evaluate_net("lenet5", sh, g, lambda v: np.zeros(len(v), dtype=int))
    assert not r["all_ok"] and r["pred_match"] == int((g["pred"] == 0).sum())
    assert r["first_mismatch_img"] == int(np.flatnonzero(g["pred"] != 0)[0])


def test_nonzero_unused_logit_is_mismatch(tmp_path):
    g = gold(10)
    text = make_log("lenet5", 0, 10, g).replace(
        result_line("lenet5", 4, g["v"][4]),
        result_line("lenet5", 4, list(g["v"][4]) + [0, 0, 0, 0, 0, 9]))
    write_shard(tmp_path, "lenet5", 0, 10, text)
    sh = [cf.load_shard(tmp_path / "lenet5" / cf.shard_dir_name(0, 10), "lenet5", 0, 10)]
    r = cf.evaluate_net("lenet5", sh, g, argmax_predict)
    assert not r["all_ok"] and r["first_mismatch_img"] == 4


def test_incomplete_shard_fails_net(tmp_path):
    g = gold(20)
    sh = shards_for(tmp_path, "lenet5", [(0, 10), (10, 20)], g, **{"10": {"done": False}})
    r = cf.evaluate_net("lenet5", sh, g, argmax_predict)
    assert not r["all_ok"] and r["shards_complete"] == 1 and r["images"] == 20
    sh2 = shards_for(tmp_path / "b", "lenet5", [(0, 10), (10, 20)], g, **{"10": {"skip": (13,)}})
    r2 = cf.evaluate_net("lenet5", sh2, g, argmax_predict)
    assert not r2["all_ok"] and r2["images"] == 19 and r2["images_expected"] == 20


def test_missing_shard_dir(tmp_path):
    g = gold(20)
    sh = shards_for(tmp_path, "lenet5", [(0, 10)], g)
    sh.append(cf.load_shard(tmp_path / "lenet5" / cf.shard_dir_name(10, 20), "lenet5", 10, 20))
    r = cf.evaluate_net("lenet5", sh, g, argmax_predict)
    assert not r["all_ok"] and r["images"] == 10 and r["images_expected"] == 20


# --------------------------------------------------------------------------- xval helpers
def test_compare_runs_and_layer_cycles():
    g = gold(4)
    a = cf.parse_log(make_log("lenet5", 0, 4, g, sim="xsim"))["results"]
    b = cf.parse_log(make_log("lenet5", 0, 4, g))["results"]
    assert cf.compare_runs(a, b) == []
    b2 = cf.parse_log(make_log("lenet5", 0, 4, g, bad_logit=(2,)))["results"]
    errs = cf.compare_runs(a, b2)
    assert len(errs) == 1 and errs[0].startswith("img 2 logits")
    assert cf.same_layer_cycles("1,2,3,4", "1,2,3,4,0")
    assert not cf.same_layer_cycles("1,2,3,4", "1,2,3,5")


# --------------------------------------------------------------------------- end-to-end collect
def test_collect_cli(tmp_path):
    g = gold(20, seed=3)
    run, data = tmp_path / "run", tmp_path / "data"
    (data / "lenet5").mkdir(parents=True)
    np.savez(data / "lenet5" / "golden.npz", **g)
    (data / "lenet5" / "meta.json").write_text(json.dumps({"git_commit": "abc"}))
    write_plan(run, "lenet5", [(0, 10), (10, 20)])
    shards_for(run, "lenet5", [(0, 10), (10, 20)], g)
    out = tmp_path / "out.csv"
    shard_csv = tmp_path / "shards.csv"
    # predictions use the real final_layer dequant (lenet5 S_a/S_w): compare against those
    import final_layer
    import gos_golden as gg
    G = gg.load_net("lenet5")
    g["pred"] = final_layer.predict_from_raw(g["v"], {"S_a": G.final.S_a, "S_w": G.final.S_w})
    np.savez(data / "lenet5" / "golden.npz", **g)
    rc = cf.main(["collect", "--run-dir", str(run), "--data-dir", str(data), "--nets", "lenet5",
                  "--csv", str(out), "--shard-csv", str(shard_csv)])
    assert rc == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == "rtl_sim" and r["net"] == "lenet5" and r["simulator"] == "verilator"
    assert r["images"] == "20" and r["logits_bitexact"] == "20" and r["all_ok"] == "True"
    assert r["git_dirty"] in ("True", "False") and r["num_inferences"] == "20"
    assert len(list(csv.DictReader(shard_csv.open()))) == 2
    # break one shard -> nonzero exit, row says not ok
    write_shard(run, "lenet5", 10, 20, make_log("lenet5", 10, 20, g, done=False))
    assert cf.main(["collect", "--run-dir", str(run), "--data-dir", str(data), "--nets", "lenet5",
                    "--csv", str(out)]) == 1
    assert list(csv.DictReader(out.open()))[0]["all_ok"] == "False"


# ---- log archive ------------------------------------------------------------------------------
def test_archive_deterministic_excludes_obj_and_lists_records(tmp_path):
    import tarfile
    run = tmp_path / "runs" / "abc123" / "verilator"
    (run / "lenet5" / "shard_00000_00500").mkdir(parents=True)
    (run / "lenet5" / "shard_00000_00500" / "run.log").write_text("RESULT img=0\nSHARD_DONE\n")
    (run / "lenet5" / "shard_00000_00500" / "status").write_text("0 1 2\n")
    (run / "build" / "obj").mkdir(parents=True)
    (run / "build" / "obj" / "Vtb.o").write_bytes(b"\0" * 64)
    (run / "build" / "build.log").write_text("ok\n")
    (run / "shards.csv").write_text("net\nlenet5\n")
    out1, out2 = tmp_path / "res" / "rtl_full10k_logs_abc123.tar.xz", tmp_path / "a2.tar.xz"
    h1 = cf.write_archive(run.parent, out1)
    h2 = cf.write_archive(run.parent, out2)
    assert h1 == h2                                    # deterministic bytes
    with tarfile.open(out1, "r:xz") as tf:
        names = tf.getnames()
    assert "runs/abc123/verilator/lenet5/shard_00000_00500/run.log" in names
    assert "runs/abc123/verilator/shards.csv" in names and not any("/obj/" in n for n in names)
    old = tmp_path / "res" / "rtl_full10k_logs_old999.tar.xz"
    old.write_bytes(b"x")
    rec = cf.record_archives(out1)
    assert rec.startswith("rtl_full10k_logs_old999.tar.xz:") and "abc123" not in rec
