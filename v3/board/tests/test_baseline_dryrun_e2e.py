"""End-to-end DRY RUN of the baseline flow on the laptop: throwaway checkpoint -> ONNX export (real ORT
quantization) + a fake DPU build -> make_baseline_package -> baseline_session --backend model (sessions 1, 2,
resume) -> aggregate_sessions. Needs torch + onnxruntime (repo .venv) and data/raw CIFAR-10."""
import csv
import json

import numpy as np
import pytest

import baseline_common as bc

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
import laptop_common as lc  # noqa: E402

if not (lc.CIFAR_DIR / "test_batch").is_file():
    pytest.skip("CIFAR-10 python batches not in data/raw", allow_module_level=True)

NET = "resnet20_b"


def _fake_dpu_build(root, ci, n_test):
    """What export_data.py + run_docker.sh would leave in v3/dpu/build/ (xmodel = dummy bytes)."""
    nd = root / NET
    (nd / "compiled").mkdir(parents=True)
    keep = {k: ci[k] for k in ("checkpoint", "checkpoint_sha256", "checkpoint_kind", "preprocess")}
    (nd / "export_info.json").write_text(json.dumps({**keep, "fp32_correct": 0}))
    rng = np.random.default_rng(0)
    np.save(nd / "vaiq_pred.npy", rng.integers(0, 10, n_test))
    (nd / "quant_info_test.json").write_text(json.dumps({"checkpoint_sha256": ci["checkpoint_sha256"],
                                                         "vaiq_correct": 1000, "vaiq_total": n_test}))
    xm = nd / "compiled" / f"{NET}_kv260.xmodel"
    xm.write_bytes(b"not a real xmodel (test)")
    (nd / "compiled" / "xmodel_info.json").write_text(json.dumps({
        "xmodel": xm.name, "xmodel_sha256": bc.sha256_file(xm), "fingerprint_matches_overlay": True,
        "overlay_fingerprint": "0x101000016010407", "dpu_fingerprint_hex": ["0x101000016010407"], "n_subgraphs": 3,
        "n_dpu_subgraphs": 1, "n_cpu_subgraphs": 1, "cpu_compute_ops": [], "all_compute_on_dpu": True,
        "dpu_input": [{"dims": [1, 32, 32, 3], "dtype": "xint8", "fix_point": 6}],
        "dpu_output": [{"dims": [1, 10], "dtype": "xint8", "fix_point": 2}]}))
    (root / "overlay_arch_info.json").write_text(json.dumps({"arch": {"fingerprint": "0x101000016010407"}}))
    link = root / "pynq_dpu_2.5" / "pynq_dpu-2.5" / "pynq_dpu"
    link.mkdir(parents=True)
    for f in ("dpu.bit.link", "dpu.hwh.link"):
        (link / f).write_text(json.dumps({"KV260": {"url": "x", "md5sum": "0" * 32}}))


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    import torch
    import nets
    import export_onnx
    import make_baseline_package as mbp
    t = tmp_path_factory.mktemp("bl")
    torch.manual_seed(0)
    m = nets.build(NET).eval()
    ck = t / "ckpt.pt"
    torch.save({"net": NET, "state_dict": m.state_dict(), "dryrun_throwaway": True}, ck)
    _, ci = lc.load_checkpoint(NET, ck)
    assert ci["checkpoint_kind"] == bc.CHECKPOINT_THROWAWAY
    export_onnx.export(NET, str(ck), limit=40, out_root=t / "onnx")
    _fake_dpu_build(t / "dpu", ci, 10000)
    pkg = t / "package"
    rc = mbp.main(["--nets", NET, "--out", str(pkg), "--dpu-build", str(t / "dpu"), "--onnx-root", str(t / "onnx"),
                   "--allow-dirty", "--allow-throwaway"])
    assert rc == 0
    bc.verify_manifest(pkg)
    return t, pkg


def _session(pkg, rd, k, *extra):
    import baseline_session as bs
    return bs.main([str(k), "--backend", "model", "--pkg-dir", str(pkg), "--results-dir", str(rd), "--limit", "30",
                    "--warmup", "2", "--phase-s", "0.25", "--threads", "1", "2", "--meas-cores", "0",
                    "--cpun-cores", "0-1", "--progress", "0", *extra])


def test_package_refuses_throwaway_without_flag(package, tmp_path):
    import make_baseline_package as mbp
    t, _ = package
    with pytest.raises(SystemExit):
        mbp.main(["--nets", NET, "--out", str(tmp_path / "p"), "--dpu-build", str(t / "dpu"),
                  "--onnx-root", str(t / "onnx"), "--allow-dirty"])


def test_board_backend_refused_on_laptop(package):
    import platform
    import baseline_session as bs
    if platform.machine() in ("aarch64", "arm64"):
        pytest.skip("on an aarch64 host")
    assert bs.main(["1", "--pkg-dir", str(package[1])]) == 2


def test_dryrun_sessions_resume_and_aggregate(package):
    import aggregate_sessions as agg
    t, pkg = package
    rd = t / "dryrun" / "baseline"
    assert _session(pkg, rd, 1) == 0
    acc = list(csv.DictReader(open(rd / "hw_baseline_accuracy.csv")))
    lat = list(csv.DictReader(open(rd / "hw_baseline_latency.csv")))
    assert [r["config"] for r in acc] == ["dpu", "ort_int8_t1", "ort_int8_t2", "ort_fp32_t1", "ort_fp32_t2"]
    assert len(lat) == 5 * 4
    for r in acc + lat:
        assert r["source"] == bc.SOURCE_DRYRUN and r["paper_grade"] == "False" and r["session_index"] == "1"
        assert r["checkpoint_kind"] == bc.CHECKPOINT_THROWAWAY
    by = {r["config"]: r for r in acc}
    assert by["dpu"]["agree_ref"] == "30"                 # fake runner replays the vai_q predictions
    assert by["ort_fp32_t1"]["agree_ref"] == "30"         # ORT FP32 == PyTorch FP32 predictions
    assert by["ort_int8_t2"]["cpu_affinity"] == "0-1" and by["dpu"]["cpu_affinity"] == "0"
    e2e = [r for r in lat if r["metric"] == "e2e"]
    for r in e2e:
        assert float(r["median_ci_lo_us"]) <= float(r["median"]) <= float(r["median_ci_hi_us"])
        assert float(r["p50"]) <= float(r["p95"]) <= float(r["p99"]) <= float(r["max"])
    summ = list(csv.DictReader(open(rd / f"hw_baseline_power_ina260_summary_{NET}.csv")))
    assert {r["config"] for r in summ if r["row_kind"] == "mean"} == set(by)
    info = json.loads((rd / "hw_baseline_session_info.json").read_text())
    assert all(s["status"] == "ok" for s in info["steps"].values())
    # resume: every step skipped, outputs unchanged
    before = {p.name: bc.sha256_file(p) for p in rd.glob("hw_baseline_preds_*.npz")}
    assert _session(pkg, rd, 1) == 0
    assert before == {p.name: bc.sha256_file(p) for p in rd.glob("hw_baseline_preds_*.npz")}
    # provenance change (other session parameters are fine; a different package is not) -> exit 4
    assert _session(pkg, rd, 2) == 0
    rows, src, probs = agg.aggregate(rd, (1, 2))
    assert src == bc.SOURCE_DRYRUN and not probs
    acc_g = [r for r in rows if r["metric_file"] == "hw_baseline_accuracy.csv" and r["metric"] == "accuracy_pct"]
    assert len(acc_g) == 5 and all(r["n_sessions"] == 2 for r in acc_g)
