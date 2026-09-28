"""Tests of v2/scripts/verification_stats.py on synthetic results CSVs (no simulator, no Vivado)."""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import verification_stats as vs  # noqa: E402
from common import META_COLUMNS  # noqa: E402  (v2/model on sys.path via verification_stats)

C1, C2 = "a" * 40, "b" * 40


def write(d: Path, name: str, rows: list[dict]):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (d / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def meta(commit=C1, dirty=False, **kw):
    return dict(git_commit=commit, git_dirty=str(dirty), **kw)


@pytest.fixture
def rd(tmp_path):
    write(tmp_path, "unit_tb.csv", [
        meta(tb="tb_a", checks="10", passed="True", duration_s="1.5", source="rtl_sim"),
        meta(tb="tb_gos_top_backtoback", checks="7", passed="True", duration_s="2", source="rtl_sim"),
        meta(C2, True, tb="tb_b", checks="100", passed="False", duration_s="4", source="rtl_sim"),
    ])
    lay = dict(model_cycles="5", rtl_cycles="5", model_total="8", rtl_total="8", model_mac="4", rtl_mac="4",
               out_ok="True", duration_s="3")
    write(tmp_path, "rtl_cycles.csv", [
        meta(kind="layer", **lay), meta(kind="layer", **{**lay, "rtl_cycles": "6"}),
        meta(kind="fuzz", **lay), meta(kind="fuzz", **{**lay, "out_ok": "False", "duration_s": "9"}),
        meta(kind="fuzz", **lay),
    ])
    net = dict(logits_match="True", pred_match="True", cycles_ok="True", model_total="9", rtl_total="9",
               duration_s="6")
    write(tmp_path, "rtl_network.csv", [meta(net="lenet5", **net), meta(net="lenet5", **net),
                                         meta(net="cifar10", **{**net, "logits_match": "False"})])
    write(tmp_path, "rtl_checker.csv", [meta(ok="True", accepted="True", duration_s="1"),
                                         meta(ok="True", accepted="False", duration_s="1")])
    write(tmp_path, "rtl_netlist.csv", [meta(tb="tb_gos_top", checks="5", passed="True", jobs_completed="2",
                                             cycles_exact="2", logits_checked="2", logits_ok="2",
                                             soft_resets="0")])
    write(tmp_path, "requant_equivalence.csv", [
        meta(net="lenet5", B="32", values_checked_exact="10", values_checked_saturated="5", mismatches="0",
             s_in_range="True"),
        meta(net="lenet5", B="32", values_checked_exact="1", values_checked_saturated="1", mismatches="0",
             s_in_range="True"),
        meta(net="lenet5", B="48", values_checked_exact="0", values_checked_saturated="0", mismatches="0",
             s_in_range="False"),
    ])
    write(tmp_path, "final_layer_check.csv", [meta(net="lenet5", images="100", logits_bitexact="100",
                                                   argmax_match="99")])
    write(tmp_path, "golden_crosscheck.csv", [
        meta(net="lenet5", images="100", bit_identical="100", predictions_identical="100"),
        meta(net="lenet5", images="100", bit_identical="98", predictions_identical="100")])
    write(tmp_path, "reference_accuracy.csv", [meta(net="lenet5", reference_version="lenet5_v1",
                                                    precision="INT8", total="100", correct="97",
                                                    accuracy_pct="97.0")])
    return tmp_path


def get(stats, activity, metric):
    hits = [s for s in stats if s["activity"] == activity and s["metric"] == metric]
    assert len(hits) == 1, (activity, metric, hits)
    return hits[0]


def test_unit_tbs_clean_and_dirty_separated(rd):
    st = vs.collect(rd, with_pytest=False)
    s = get(st, "unit TBs", "total checks")
    assert (s["value"], s["value_dirty"], s["rows_used"], s["rows_dirty"]) == (17, 100, 2, 1)
    assert s["flag"] == "DIRTY" and s["git_commits"] == f"{C1[:8]};{C2[:8]}"
    assert get(st, "unit TBs", "testbenches")["value"] == 2
    assert get(st, "unit TBs", "all passed")["value"] == "True"   # clean rows only
    # a failing TB (even a dirty one) raises FAIL
    assert "FAIL" in get(st, "unit TBs", "all passed")["flag"]
    assert get(st, "unit TBs", "recorded runtime s (sum of duration_s)")["value"] == 3.5
    assert get(st, "back-to-back jobs (RTL)", "checks")["value"] == 7
    assert get(st, "back-to-back jobs (RTL)", "jobs")["flag"].startswith("NOT_RECORDED")


def test_core_suites(rd):
    st = vs.collect(rd, with_pytest=False)
    assert get(st, "core layer suite", "layer runs")["value"] == 2
    assert get(st, "core layer suite", "cycle matches (layer, total, MAC_ACTIVE = model)")["value"] == 1
    assert get(st, "core fuzz suite", "fuzz shapes")["value"] == 3
    assert get(st, "core fuzz suite", "output matches")["value"] == 2
    assert get(st, "core fuzz suite", "runtime s (duration_s)")["value"] == 9
    assert get(st, "core net suite", "lenet5: images")["value"] == 2
    assert get(st, "core net suite", "cifar10: logit matches")["value"] == 0
    assert get(st, "core checker suite", "refused cases")["value"] == 1
    for s in st:
        if s["source_csv"].startswith("rtl_") and s["source_csv"] != "rtl_netlist.csv":
            assert s["label"] == "rtl_sim"
    assert get(st, "netlist sim (post-synthesis funcsim)", "tb_gos_top: logit-checked jobs ok / checked")[
        "value"] == "2/2"
    assert get(st, "netlist sim (post-synthesis funcsim)", "tb_gos_top: passed")["label"] == "post_synth"


def test_model_checks(rd):
    st = vs.collect(rd, with_pytest=False)
    assert get(st, "requant equivalence", "lenet5 B=32 (adopted, hw_requant.npz): values checked")["value"] == 17
    assert get(st, "requant equivalence", "lenet5 B=48: channels with s out of range")["value"] == 1
    assert get(st, "final-layer check", "lenet5: argmax matches")["value"] == 99
    g = "golden cross-check (gos_golden vs legacy)"
    assert get(st, g, "lenet5: tensors bit-identical on all images")["value"] == 1
    assert get(st, "reference accuracy", "lenet5_v1 INT8: correct")["value"] == 97
    assert all(s["label"] == "model" for s in st if s["source_csv"] in
               ("requant_equivalence.csv", "final_layer_check.csv", "reference_accuracy.csv"))


def test_board_a1_pending_then_hw_only(rd):
    st = vs.collect(rd, with_pytest=False)
    s = get(st, "board A1 (KV260)", "images")
    assert s["value"] == "pending" and s["flag"] == "PENDING"
    write(rd, "hw_a1_accuracy.csv", [
        meta(net="lenet5", source="hw", images="10000", logit_mismatch_images="0", pred_mismatches="0",
             job_errors="0", hw_correct="9000"),
        meta(net="lenet5", source="dryrun_model", images="5", logit_mismatch_images="0", pred_mismatches="0",
             job_errors="0", hw_correct="5")])
    st = vs.collect(rd, with_pytest=False)
    assert get(st, "board A1 (KV260)", "lenet5: images")["value"] == 10000
    assert get(st, "board A1 (KV260)", "lenet5: images")["label"] == "hw"
    assert get(st, "board A1 (KV260)", "non-hw rows ignored")["value"] == 1


def test_missing_csv_is_pending(tmp_path):
    st = vs.collect(tmp_path, with_pytest=False)
    assert get(st, "unit TBs", "rows")["flag"] == "PENDING"
    assert get(st, "core net suite", "rows")["value"] == "missing"


def test_pytest_count_parsing(monkeypatch):
    class R:
        returncode, stdout = 0, "a::b\n\n42 tests collected in 1.0s\n"
    monkeypatch.setattr(vs.subprocess, "run", lambda *a, **k: R())
    st = vs.Stats()
    vs.pytest_count(st, paths=("v2/model",))
    assert st.rows[0]["value"] == 42 and st.rows[0]["label"] == "model"


def test_main_writes_csv_with_metadata(rd, tmp_path, capsys):
    out = tmp_path / "o" / "verification_stats.csv"
    rc = vs.main(["--results-dir", str(rd), "--out", str(out), "--no-pytest"])
    assert rc == 1  # the synthetic unit_tb.csv has a failing TB
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0])[:len(META_COLUMNS)] == list(META_COLUMNS)
    assert set(vs.FIELDS) <= set(rows[0])
    assert all(r["source"] == r["label"] for r in rows)
    assert "| Activity | Metric | Value |" in capsys.readouterr().out


def test_real_results_run(tmp_path):
    """The script runs on the repository's results CSVs (read-only) and writes elsewhere."""
    out = tmp_path / "vs.csv"
    r = subprocess.run([sys.executable, str(SCRIPTS / "verification_stats.py"), "--no-pytest", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr
    assert out.exists() and "verification_stats: wrote" in r.stdout
