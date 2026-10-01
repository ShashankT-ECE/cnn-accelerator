"""dpu_session.py without hardware: argument handling, package checks, I/O conversion, dry run.

    .venv/bin/python -m pytest v2/dpu/tests -q

The dry-run end-to-end test needs the board data package v2/board/data/<net>/ (built by
v2/board/make_board_data.py, gitignored); it is skipped when absent. The DPU package used here is
FAKE (dummy xmodel bytes): it tests the plumbing only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DPU = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DPU))
import dpu_session as ds  # noqa: E402

DATA = DPU.parent / "board" / "data"
SHAPES = {"lenet5": (32, 32, 1), "cifar10": (32, 32, 3)}


def fake_pkg(root: Path, nets=("lenet5", "cifar10"), n_pred=10000, dirty=False) -> Path:
    info = {"git_commit": "f" * 40, "git_dirty": dirty, "vitis_ai_version": "2.5.0",
            "docker_image": "xilinx/vitis-ai-cpu:2.5.0", "dpu_arch": "DPUCZDX8G_ISA1_B4096",
            "quantizer": "fake", "nets": {}}
    for n in nets:
        d = root / n
        d.mkdir(parents=True)
        xm = d / f"{n}_kv260.xmodel"
        xm.write_bytes(b"not a real xmodel " + n.encode())
        info["nets"][n] = {"xmodel": xm.name,
                           "xmodel_sha256": hashlib.sha256(xm.read_bytes()).hexdigest()}
        xi = {"dpu_fingerprint_hex": ["0x101000016010407"], "n_dpu_subgraphs": 1, "n_cpu_subgraphs": 0,
              "dpu_input": [{"dims": [1, *SHAPES[n]], "dtype": "xint8", "fix_point": 6}],
              "dpu_output": [{"dims": [1, 10], "dtype": "xint8", "fix_point": 2}]}
        (d / "xmodel_info.json").write_text(json.dumps(xi))
        np.save(d / "vaiq_pred.npy", np.arange(n_pred, dtype=np.int64) % 10)
    (root / ds.PACKAGE_INFO).write_text(json.dumps(info))
    return root


# ---- unit ----------------------------------------------------------------------------------
def test_fill_input_int8_round_half_up_and_clip():
    io = ds.IoSpec([1, 2, 2, 1], "xint8", 7, [1, 10], "xint8", 3)
    buf = np.zeros((1, 2, 2, 1), np.int8)
    x = np.array([[[0.0], [1.0]], [[0.5 / 128], [0.25]]], np.float32)   # 0, 128->127, 0.5->1, 32
    ds.fill_input(buf, x, io)
    assert buf[0].reshape(-1).tolist() == [0, 127, 1, 32]
    assert io.in_scale == 128.0 and io.out_scale == 0.125


def test_float_io_copies():
    io = ds.IoSpec([1, 2, 2, 1], "float32", None, [1, 10], "float32", None)
    inp, out = io.buffers()
    assert inp[0].dtype == np.float32 and out[0].dtype == np.float32
    x = np.full((2, 2, 1), 0.3, np.float32)
    ds.fill_input(inp[0], x, io)
    assert np.array_equal(inp[0][0], x)


def test_batch_must_be_one():
    with pytest.raises(SystemExit):
        ds.IoSpec([4, 32, 32, 1], "xint8", 6, [4, 10], "xint8", 2)


def test_to_nhwc():
    x = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    y = ds.to_nhwc(x)
    assert y.shape == (2, 4, 5, 3) and y.flags.c_contiguous and y[1, 2, 3, 0] == x[1, 0, 2, 3]


def test_run_images_fake_runner_preds_and_timing():
    io = ds.IoSpec([1, 4, 4, 1], "xint8", 6, [1, 10], "xint8", 2)
    preds = np.array([3, 1, 4, 1, 5], np.int64)
    r = ds.FakeRunner(io, preds)
    x = np.zeros((5, 4, 4, 1), np.float32)
    p, raw, t = ds.run_images(r, io, x, range(5))
    assert p.tolist() == preds.tolist() and raw.shape == (5, 10)
    assert set(t) == {"pre", "dpu_runner", "post", "end_to_end"}
    assert (t["end_to_end"] >= t["dpu_runner"]).all()


def test_xmodel_sha_mismatch_refused(tmp_path):
    root = fake_pkg(tmp_path / "pkg", nets=("lenet5",))
    (root / "lenet5" / "lenet5_kv260.xmodel").write_bytes(b"edited")
    with pytest.raises(SystemExit, match="sha256"):
        ds.load_dpu_package(root, ["lenet5"])


def test_missing_package_info(tmp_path):
    with pytest.raises(SystemExit, match="DPU_INFO"):
        ds.load_dpu_package(tmp_path, ["lenet5"])


@pytest.mark.parametrize("bad", [["--limit", "0"], ["--repeats", "0"], ["--phase-s", "0"],
                                 ["--nets", "resnet"], ["--no-power", "--power-only"]])
def test_bad_args(bad):
    with pytest.raises(SystemExit):
        ds.parse(["--dry-run", *bad])


def test_dry_defaults():
    a = ds.parse(["--dry-run"])
    assert a.phase_s == 1.0 and a.repeats == 1 and a.warmup == ds.DEFAULT_WARMUP
    b = ds.parse([])
    assert b.phase_s == 60.0 and b.repeats == 3


def test_dry_run_refuses_non_dryrun_out_dir(tmp_path):
    root = fake_pkg(tmp_path / "pkg", nets=("lenet5",))
    if not (DATA / "lenet5" / "MANIFEST.json").is_file():
        pytest.skip("board data package not built")
    with pytest.raises(SystemExit, match="dryrun"):
        ds.main(["--dry-run", "--pkg-dir", str(root), "--nets", "lenet5", "--limit", "5",
                 "--no-power", "--out-dir", str(tmp_path / "results")])


# ---- end-to-end dry run --------------------------------------------------------------------
@pytest.mark.skipif(not all((DATA / n / "MANIFEST.json").is_file() for n in ("lenet5", "cifar10")),
                    reason="board data package not built")
def test_dry_run_end_to_end(tmp_path):
    root = fake_pkg(tmp_path / "pkg")
    out = tmp_path / "dryrun" / "dpu"
    rc = ds.main(["--dry-run", "--pkg-dir", str(root), "--limit", "120", "--warmup", "5",
                  "--phase-s", "0.3", "--repeats", "2", "--rate-hz", "20", "--out-dir", str(out)])
    assert rc == 0
    acc = list(csv.DictReader((out / "hw_dpu_accuracy.csv").open()))
    assert [r["net"] for r in acc] == ["lenet5", "cifar10"]
    for r in acc:
        assert r["source"] == "dryrun_model" and r["images"] == "120" and r["batch"] == "1"
        assert r["measurement"].startswith("DPU (Vitis AI 2.5.0, DPUCZDX8G_ISA1_B4096, fingerprint")
        assert r["agree_vaiq_model"] == "120"        # fake runner replays vaiq_pred
    lat = list(csv.DictReader((out / "hw_dpu_latency.csv").open()))
    assert {(r["net"], r["metric"]) for r in lat} == {
        (n, m) for n in ("lenet5", "cifar10") for m in ("dpu_runner", "pre", "post", "end_to_end")}
    for n in ("lenet5", "cifar10"):
        z = np.load(out / f"hw_dpu_preds_{n}.npz")
        assert z["pred"].shape == (120,) and z["t_end_to_end_ns"].shape == (120,)
        ph = list(csv.DictReader((out / f"hw_dpu_power_ina260_phases_{n}.csv").open()))
        assert [p["phase"] for p in ph] == ["idle_pre", "accel", "idle_mid"] * 2
        assert all(p["measurement"] == "SOM-rail power (INA260)" for p in ph)
        assert all(p["source"] == "dryrun_model" for p in ph)
        sm = list(csv.DictReader((out / f"hw_dpu_power_ina260_summary_{n}.csv").open()))
        assert [s["row_kind"] for s in sm] == ["repeat", "repeat", "mean", "std"]
        assert sm[0]["accel_workload"].startswith("DPU (Vitis AI 2.5.0")
    info = json.loads((out / "hw_dpu_session_info.json").read_text())
    assert info["source"] == "dryrun_model"


@pytest.mark.skipif(not all((DATA / n / "MANIFEST.json").is_file() for n in ("lenet5", "cifar10")),
                    reason="board data package not built")
def test_rows_carry_the_run_environment(tmp_path, monkeypatch):
    """Rows of the orchestrated step carry session_index / env_step / paper_grade columns ($GOS_RUN_ENV).
    (The 2026-09-30 DPU rows had them empty -> rejected by the paper tables.) Dry run: paper_grade False."""
    monkeypatch.setenv("GOS_RUN_ENV", json.dumps({"session_index": 2, "paper_grade": True, "step_id": "s2.DPU",
                                                  "cpu_governor": "performance", "cpu_freq_khz": 1333333}))
    root = fake_pkg(tmp_path / "pkg")
    out = tmp_path / "dryrun" / "dpu"
    assert ds.main(["--dry-run", "--pkg-dir", str(root), "--nets", "lenet5", "--limit", "20", "--warmup", "2",
                    "--phase-s", "0.2", "--repeats", "1", "--rate-hz", "20", "--out-dir", str(out)]) == 0
    for f in ("hw_dpu_accuracy.csv", "hw_dpu_latency.csv", "hw_dpu_power_ina260_phases_lenet5.csv"):
        rows = list(csv.DictReader((out / f).open()))
        assert rows and all(r["session_index"] == "2" and r["env_step"] == "s2.DPU"
                            and r["cpu_governor"] == "performance" and r["paper_grade"] == "False" for r in rows), f
