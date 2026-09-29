"""exp_layer_spread: statistics, npz reuse rules, RTL-sim log parsing, dry run, session step."""
import csv

import numpy as np
import pytest

import board_common as bc
import exp_layer_spread as L
import session_extra_steps as X


def test_spread_stats():
    s = L.spread_stats(np.array([5, 5, 5]), 5)
    assert (s["cyc_min"], s["cyc_max"], s["cyc_distinct"], s["cyc_spread"], s["images_equal_model"],
            s["result"]) == (5, 5, 1, 0, 3, "PASS")
    s = L.spread_stats(np.array([5, 6, 5]), 5)
    assert s["cyc_distinct"] == 2 and s["cyc_spread"] == 1 and s["images_equal_model"] == 2 and s["result"] == "FAIL"
    assert L.spread_stats(np.array([7, 7]), 5)["result"] == "FAIL"          # constant but != model
    assert L.spread_stats(np.array([], np.int64), 5)["result"] == "FAIL"


def test_layer_table():
    lc = np.array([[1, 2], [1, 2], [1, 3]])
    t = L.layer_table(["a", "b"], [1, 2], 4, lc, np.array([4, 4, 5]))
    assert [r["layer"] for r in t] == ["a", "b", "total"]
    assert [r["result"] for r in t] == ["PASS", "FAIL", "FAIL"]


def _npz(p, n=4, source="dryrun_model", clk=200.0, ok=None, idx=None, layer=True):
    d = dict(idx=np.arange(n) if idx is None else idx, total_cyc=np.full(n, 9),
             ok=np.ones(n, bool) if ok is None else ok, clock_mhz=clk, source=np.array(source))
    if layer:
        d["layer_cyc"] = np.tile([4, 5], (n, 1))
    np.savez(p, **d)


def test_npz_reuse_rules(tmp_path):
    f = lambda: L.load_npz_counters(tmp_path, "lenet5", "dryrun_model", 4, 200.0)  # noqa: E731
    got, p, why = f()
    assert got is None and "absent" in why
    _npz(tmp_path / "hw_cycles_lenet5.npz", source="hw")
    assert f()[0] is None and "source" in f()[2]
    _npz(tmp_path / "hw_cycles_lenet5.npz", n=3)
    assert f()[0] is None and "covers 3" in f()[2]
    _npz(tmp_path / "hw_cycles_lenet5.npz", ok=np.array([1, 1, 0, 1], bool))
    assert f()[0] is None and "failed jobs" in f()[2]
    _npz(tmp_path / "hw_cycles_lenet5.npz", clk=250.0)
    assert f()[0] is None and "clock" in f()[2]
    _npz(tmp_path / "hw_cycles_lenet5.npz", layer=False)
    assert f()[0] is None and "layer_cyc" in f()[2]
    # A1's npz is accepted as the fallback when it carries layer_cyc
    _npz(tmp_path / "hw_logits_lenet5.npz", idx=np.array([3, 1, 2, 0]))
    got, p, why = f()
    assert got is not None and p.name == "hw_logits_lenet5.npz" and got["layer_cyc"].shape == (4, 2)
    _npz(tmp_path / "hw_cycles_lenet5.npz", clk=200.3)                  # within tolerance, preferred
    assert f()[1].name == "hw_cycles_lenet5.npz"


def test_dry_run_reuses_then_reruns(tmp_path):
    out = tmp_path / "dryrun"
    rc = L.main(["--backend", "model", "--allow-dirty", "--limit", "5", "--nets", "lenet5",
                 "--out-dir", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(open(out / "hw_layer_spread.csv")))
    assert all(r["cycles_origin"].startswith("own run") for r in rows)
    assert all(r["result"] == "PASS" and r["cyc_spread"] == "0" for r in rows)
    assert rows[-1]["layer"] == "total" and {r["source"] for r in rows} == {bc.SOURCE_DRYRUN}
    # the own run's npz is not a hw_cycles npz; supply one (A2/A3 layout) and it is reused
    d = np.load(out / "hw_layer_spread_lenet5.npz")
    np.savez(out / "hw_cycles_lenet5.npz", **{k: d[k] for k in d.files})
    rc = L.main(["--backend", "model", "--allow-dirty", "--limit", "5", "--nets", "lenet5",
                 "--out-dir", str(out / "b"), "--npz-dir", str(out)])
    rows = list(csv.DictReader(open(out / "b" / "hw_layer_spread.csv")))
    assert rc == 0 and all("reused" in r["cycles_origin"] and len(r["cycles_origin_sha256"]) == 64 for r in rows)


def _log(net, imgs, layers="2829,6029,6029,1349,197", total=16436):
    lines = ["SIMULATOR name=verilator version=5.028"]
    for i in imgs:
        lines.append(f"RESULT kind=full net={net} img={i} logits_ok=1 cycles_ok=1 rtl_total={total} "
                     f"model_total=16436 mac=1 stall=0 logits=0 layer_cycles={layers}")
    return "\n".join(lines) + "\n"


def test_rtl_spread_from_logs(tmp_path):
    for a, b in ((0, 2), (2, 4)):
        d = tmp_path / "lenet5" / f"shard_{a:05d}_{b:05d}"
        d.mkdir(parents=True)
        (d / "run.log").write_text(_log("lenet5", range(a, b)))
    got, status, shards, n = L.rtl_spread(tmp_path, "lenet5", 4)
    assert status == "ok" and shards == 2 and n == 4 and got[0].shape == (4, 5)
    got, status, _, _ = L.rtl_spread(tmp_path, "lenet5", 10)
    assert status.startswith("partial")
    got, status, _, _ = L.rtl_spread(tmp_path, "cifar10", 10)
    assert got is None and "not present" in status


def test_rtl_mode_output_rules(tmp_path):
    with pytest.raises(SystemExit):
        L.main(["--rtl-sim", "--run-dir", str(tmp_path), "--out-dir", str(tmp_path / "res")])
    rc = L.main(["--rtl-sim", "--run-dir", str(tmp_path / "missing"), "--out-dir", str(tmp_path / "dryrun")])
    rows = list(csv.DictReader(open(tmp_path / "dryrun" / "rtl_layer_spread.csv")))
    assert rc == 0 and len(rows) == 2 and all("not present" in r["status"] and r["result"] == "N/A" for r in rows)
    assert {r["source"] for r in rows} == {"rtl_sim"}


def test_session_spread_step():
    st = X.spread_step({"backend": "pynq", "bit": "bit/gos_300.bit", "results_dir": "results",
                        "data_dir": "data"})
    assert st["id"] == "s2.layerspread" and st["session"] == 2 and st["group"] == "A3"
    assert st["cmd"][0] == "exp_layer_spread.py" and "--npz-dir" in st["cmd"] and "--limit" not in st["cmd"]
    assert st["outputs"] == ["hw_layer_spread.csv"] and st["requires"] == []
    q = X.spread_step({"backend": "model", "quick": True, "closed_mhz": 200.0})
    assert q["cmd"][q["cmd"].index("--limit") + 1] == "200" and q["params"]["limit"] == 200
