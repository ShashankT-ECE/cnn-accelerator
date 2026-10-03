"""CSV schema / output rules / preprocessing / power protocol arithmetic of the V3 baseline package."""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import baseline_common as bc
import baseline_session as bs
import power_log as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
import common  # noqa: E402  v3/model/common.py


def test_meta_columns_match_v3_model_common():
    assert bc.META_COLUMNS == common.META_COLUMNS


def test_write_csv_column_order_and_source_guard(tmp_path):
    d = tmp_path / "dryrun"
    rows = [{"source": bc.SOURCE_DRYRUN, "net": "resnet20_b", "accuracy_pct": "1.00"}]
    p = bc.write_csv(d / "x.csv", rows, ["accuracy_pct"], bc.SOURCE_DRYRUN)
    hdr = next(csv.reader(p.open()))
    assert hdr[:len(bc.META_COLUMNS)] == list(bc.META_COLUMNS)
    assert hdr[len(bc.META_COLUMNS):len(bc.META_COLUMNS) + len(bc.EXTRA_META)] == list(bc.EXTRA_META)
    assert hdr[-1] == "accuracy_pct"
    with pytest.raises(SystemExit):                      # dry-run rows outside a dryrun dir
        bc.write_csv(tmp_path / "x.csv", rows, ["accuracy_pct"], bc.SOURCE_DRYRUN)
    with pytest.raises(SystemExit):                      # hardware rows inside a dryrun dir
        bc.write_csv(d / "y.csv", [{"source": bc.SOURCE_HW}], [], bc.SOURCE_HW)
    with pytest.raises(SystemExit):                      # row source != run source
        bc.write_csv(d / "z.csv", [{"source": bc.SOURCE_HW}], [], bc.SOURCE_DRYRUN)


def test_session_csv_fields_have_no_meta_collisions():
    for fields in (bs.ACC_FIELDS, bs.LAT_FIELDS, pl.SUMMARY_FIELDS, pl.PHASE_FIELDS, pl.SAMPLE_FIELDS):
        clash = set(fields) & set(bc.META_COLUMNS + bc.EXTRA_META)
        assert not clash, clash


def test_paper_grade_rule():
    env = {"paper_grade": True}
    assert bc.env_meta(bc.SOURCE_HW, False, env, bc.CHECKPOINT_TRAINED)["paper_grade"]
    assert not bc.env_meta(bc.SOURCE_HW, True, env, bc.CHECKPOINT_TRAINED)["paper_grade"]
    assert not bc.env_meta(bc.SOURCE_HW, False, env, bc.CHECKPOINT_THROWAWAY)["paper_grade"]
    assert not bc.env_meta(bc.SOURCE_DRYRUN, False, env, bc.CHECKPOINT_TRAINED)["paper_grade"]
    assert not bc.env_meta(bc.SOURCE_HW, False, {}, bc.CHECKPOINT_TRAINED)["paper_grade"]


@pytest.mark.parametrize("spec", [None, {"mean": [0.49, 0.48, 0.45], "std": [0.25, 0.24, 0.26]}])
def test_preprocess_hwc_equals_nchw_bitwise(spec):
    x = np.random.default_rng(1).integers(0, 256, size=(64, 32, 32, 3), dtype=np.uint8)
    a = bc.preprocess(x, spec)
    b = np.stack([bc.preprocess_hwc(x[i], spec) for i in range(len(x))]).transpose(0, 3, 1, 2)
    assert a.dtype == np.float32 and a.shape == (64, 3, 32, 32)
    assert np.array_equal(a.view(np.uint32), np.ascontiguousarray(b).view(np.uint32))


def test_configs_parse():
    assert bs.all_configs() == ["dpu", "ort_int8_t1", "ort_int8_t4", "ort_fp32_t1", "ort_fp32_t4"]
    assert bs.parse_config("ort_int8_t4") == ("ort_int8", 4)
    assert bs.parse_config("dpu") == ("dpu", None)
    for bad in ("ort_int8", "ort_int8_t0", "cpu", "ort_fp16_t1"):
        with pytest.raises(ValueError):
            bs.parse_config(bad)


def test_schedule_and_p_idle_rule():
    sch = pl.build_schedule(2, ["a", "b"], 5.0, 7.0, order_seed=3)
    assert len(sch) == pl.n_phases(2, 2) == 9
    assert [p["kind"] for p in sch] == ["idle", "run"] * 4 + ["idle"]
    for r in (1, 2):
        assert sorted(p["phase"] for p in sch if p["repeat"] == r and p["kind"] == "run") == ["a", "b"]
    assert sch == pl.build_schedule(2, ["a", "b"], 5.0, 7.0, order_seed=3)
    # synthetic phase means: idle drifts linearly 1.0 -> 1.4; run phases 2.0 W, 10 s, 100 images
    phases = []
    for i, p in enumerate(sch):
        w = 1.0 + 0.05 * i if p["kind"] == "idle" else 2.0
        phases.append({**p, "mean_w": w, "duration_s": 10.0, "images": 0 if p["kind"] == "idle" else 100})
    s = pl.summarize(phases, 2, ["a", "b"])
    first = s[0]
    assert first["p_idle_w"] == pytest.approx((1.0 + 1.1) / 2)       # bracketing idles (drift cancels)
    assert first["dp_w"] == pytest.approx(2.0 - 1.05)
    assert first["time_per_image_s"] == pytest.approx(0.1)
    assert first["energy_per_image_mj"] == pytest.approx(0.95 * 0.1 * 1e3)
    means = [r for r in s if r["row_kind"] == "mean"]
    assert {r["config"] for r in means} == {"a", "b"}
    assert all(not math.isnan(r["dp_w"]) for r in means)


class _Ctx:
    source = bc.SOURCE_DRYRUN

    def __init__(self, out):
        self.out_dir = out

    def meta(self, net="", config="", duration_s="", num_inferences=""):
        return {"timestamp": bc.utc_now(), "source": self.source, "net": net, "git_dirty": True,
                "duration_s": duration_s, "num_inferences": num_inferences}


def test_power_protocol_mock_writes_three_csvs(tmp_path):
    out = tmp_path / "dryrun"
    out.mkdir()
    wl = {"a": (lambda dl: pl.loop_until(lambda k: None, dl), "noop a"),
          "b": (lambda dl: pl.loop_until(lambda k: None, dl), "noop b")}
    res = pl.run_power_protocol(_Ctx(out), "resnet20_b", wl, sensor="mock", phase_s=0.3, repeats=1, rate_hz=50,
                                order_seed=1, sensor_kw={"levels_w": {"a": 2.0, "b": 3.0}})
    summ = list(csv.DictReader(open(res["files"]["summary"])))
    mean = {r["config"]: r for r in summ if r["row_kind"] == "mean"}
    assert float(mean["a"]["dp_w"]) == pytest.approx(1.0) and float(mean["b"]["dp_w"]) == pytest.approx(2.0)
    assert all(r["measurement"] == pl.LABEL for r in summ)
    phases = list(csv.DictReader(open(res["files"]["phases"])))
    assert len(phases) == pl.n_phases(1, 2)
    with pytest.raises(SystemExit):        # mock sensor in a hardware run is refused
        c = _Ctx(tmp_path)
        c.source = bc.SOURCE_HW
        pl.run_power_protocol(c, "x", wl, sensor=pl.MockSensor(), phase_s=0.1, repeats=1)
