"""V3 DPU baseline flow, laptop side (no Docker / xir / pynq_dpu needed)."""
import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import baseline_common as bc
import baseline_session as bs

DPU = Path(__file__).resolve().parents[1]
V3 = DPU.parent

# Files executed by the Vitis AI 2.5.0 container's Python 3.7 (nets.py is imported by vai_quantize.py there).
CONTAINER_PY = [DPU / "vai_quantize.py", DPU / "overlay_arch.py", DPU / "inspect_xmodel.py", DPU / "xmodel_diff.py",
                V3 / "model" / "nets.py"]


@pytest.mark.parametrize("path", CONTAINER_PY, ids=lambda p: p.name)
def test_container_scripts_parse_as_python37(path):
    ast.parse(path.read_text(), filename=str(path), feature_version=(3, 7))


def test_shell_scripts_syntax():
    for sh in ("run_docker.sh", "fetch_overlay_ref.sh"):
        subprocess.run(["bash", "-n", str(DPU / sh)], check=True)
    subprocess.run(["bash", "-n", str(V3 / "board" / "session.sh")], check=True)
    subprocess.run(["bash", "-n", str(V3 / "board" / "deploy_baseline.sh")], check=True)


def test_iospec_and_input_conversion():
    xi = {"dpu_input": [{"dims": [1, 32, 32, 3], "dtype": "xint8", "fix_point": 6}],
          "dpu_output": [{"dims": [1, 10], "dtype": "xint8", "fix_point": 2}]}
    io = bs.IoSpec.from_info(xi)
    assert io.in_int8 and io.in_scale == 64.0
    inp, out = io.buffers()
    assert inp[0].shape == (1, 32, 32, 3) and inp[0].dtype == np.int8 and out[0].shape == (1, 10)
    x = np.zeros((32, 32, 3), np.float32)
    x[0, 0] = [0.5 / 64, 1.0, 3.0]          # tie rounds up; 64 -> 64; 192 clips to 127
    x[0, 1] = [-0.5 / 64, -3.0, 1.5 / 64]   # -0.5 -> 0 (half up); -192 clips to -128; 1.5 -> 2
    bs.fill_input(inp[0], x, io)
    assert inp[0][0, 0, 0].tolist() == [1, 64, 127]
    assert inp[0][0, 0, 1].tolist() == [0, -128, 2]
    with pytest.raises(SystemExit):
        bs.IoSpec([2, 32, 32, 3], "xint8", 6, [2, 10], "xint8", 2)      # batch must be 1


def test_fake_runner_one_hot():
    io = bs.IoSpec([1, 32, 32, 3], "xint8", 6, [1, 10], "xint8", 2)
    r = bs.FakeRunner(io, np.array([3, 7]))
    inp, out = io.buffers()
    preds = []
    for _ in range(3):
        r.wait(r.execute_async(inp, out))
        preds.append(int(np.argmax(out[0])))
    assert preds == [3, 7, 3]


torch = pytest.importorskip("torch")
import laptop_common as lc  # noqa: E402
import nets  # noqa: E402


def test_load_checkpoint_formats(tmp_path):
    m = nets.build("mobilenet_s")
    sd = m.state_dict()
    torch.save(sd, tmp_path / "plain.pt")
    torch.save({"model_state_dict": {"module." + k: v for k, v in sd.items()},
                "preprocess": {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}}, tmp_path / "wrapped.pt")
    torch.save({"net": "resnet20_b", "state_dict": sd}, tmp_path / "other.pt")
    _, a = lc.load_checkpoint("mobilenet_s", tmp_path / "plain.pt")
    assert a["checkpoint_kind"] == bc.CHECKPOINT_TRAINED and a["preprocess"]["mean"] is None
    _, b = lc.load_checkpoint("mobilenet_s", tmp_path / "wrapped.pt")
    assert b["preprocess"]["mean"] == [0.5, 0.5, 0.5] and b["preprocess"]["source"].startswith("checkpoint")
    _, c = lc.load_checkpoint("mobilenet_s", tmp_path / "plain.pt", {"mean": [0.1] * 3, "std": [0.2] * 3})
    assert c["preprocess"]["source"] == "command line"
    with pytest.raises(SystemExit):
        lc.load_checkpoint("mobilenet_s", tmp_path / "other.pt")


@pytest.mark.skipif(not (lc.CIFAR_DIR / "test_batch").is_file(), reason="CIFAR-10 not in data/raw")
def test_export_data_writes_consistent_inputs(tmp_path):
    import export_data
    torch.manual_seed(0)
    m = nets.build("resnet20_b")
    ck = tmp_path / "ck.pt"
    torch.save({"state_dict": m.state_dict(), "dryrun_throwaway": True}, ck)
    info = export_data.export("resnet20_b", str(ck), build=tmp_path / "build")
    nd = tmp_path / "build" / "resnet20_b"
    z = np.load(nd / "dpu_inputs.npz")
    assert z["x_test"].shape == (10000, 3, 32, 32) and z["x_calib"].shape == (1024, 3, 32, 32)
    assert z["x_test"].dtype == np.float32 and z["y_test"].dtype == np.int64
    x_u8, y = lc.cifar_test()
    assert np.array_equal(z["x_test"][:5], bc.preprocess(x_u8[:5])) and np.array_equal(z["y_test"], y)
    assert info["checkpoint_kind"] == bc.CHECKPOINT_THROWAWAY
    assert bc.sha256_file(nd / "dpu_inputs.npz") == info["dpu_inputs_sha256"]
    assert json.loads((nd / "export_info.json").read_text())["fp32_correct"] == info["fp32_correct"]
    sd = torch.load(nd / "fp32_state_dict.pt")
    nets.build("resnet20_b").load_state_dict(sd)          # loads strictly into the container-side class
