"""board_efficiency.py / dpu_overlay_params.py on the committed results (skipped if no board data)."""
import csv
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
import _setup  # noqa: E402

import board_efficiency as BE  # noqa: E402
import dpu_overlay_params as DP  # noqa: E402

RES = _setup.RESULTS_DIR


def test_overlay_params_mac_lanes_and_unavailable_resources():
    if not DP.DEFAULT_HWH.exists():
        pytest.skip("overlay hwh not fetched (v2/dpu/fetch_overlay_ref.sh)")
    rows = DP.extract(DP.DEFAULT_HWH)
    by = {r["name"]: r for r in rows}
    lanes = int(by["ARCH_PP"]["value"]) * int(by["ARCH_ICP"]["value"]) * int(by["ARCH_OCP"]["value"])
    assert int(by["mac_lanes"]["value"]) == lanes == 2048 and by["ops_per_cycle"]["value"] == "4096"
    assert by["aclk"]["value"] == "300" and by["ap_clk_2"]["value"] == "600"
    assert all(by[f"resource_{k}"]["kind"] == "unavailable" and by[f"resource_{k}"]["value"] == ""
               for k in ("DSP48E2", "LUT", "FF", "BRAM36", "URAM"))


def test_efficiency_formulas(tmp_path):
    if not (RES / "hw_dpu_latency.csv").exists() or not (RES / "dpu_overlay_params.csv").exists():
        pytest.skip("board / DPU results not committed yet")
    rows = BE.compute(RES)
    cm = {r["net"]: int(r["macs"]) for r in csv.DictReader((RES / "cycle_model.csv").open()) if r["layer"] == "total"}
    assert {r["system"] for r in rows} >= {"ours_pl", "ours_e2e_fast", "ours_e2e_safe", "dpu_runner", "dpu_e2e"}
    for r in rows:
        ops = 2 * cm[r["net"]]
        assert int(r["ops_per_image"]) == ops
        assert float(r["gops"]) == pytest.approx(ops / (float(r["time_us"]) * 1e3), rel=1e-4)
        assert float(r["images_per_s"]) == pytest.approx(1e6 / float(r["time_us"]), rel=1e-4)
        if r["system"].startswith("ours"):
            assert r["mac_lanes"] == 64 and float(r["gops_per_dsp"]) == pytest.approx(float(r["gops"]) / int(r["dsps"]), rel=1e-3)
        if r["system"].startswith("dpu"):
            assert r["mac_lanes"] == 2048 and r["dsps"] == "" and "gops_per_dsp" not in r or r.get("gops_per_dsp", "") == ""
