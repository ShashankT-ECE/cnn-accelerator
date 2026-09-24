"""Frozen CIFAR-10 INT8 parameter sets: r2 = reference of record (Step 2.1b/2.1c,
DECISIONS D3/D9) and r1 = history (Step 2.1). Both are checked through NET_CONFIGS."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import common
import freeze_cifar10_int8 as fz
from net_config import NET_CONFIGS

KEYS = ("cifar10", "cifar10_r1")


def _dir(key):
    return NET_CONFIGS[key]["param_dir"]


def _ckpt(key):
    return NET_CONFIGS[key]["checkpoint"]


def _sums(key):
    out = {}
    for line in (_dir(key) / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        out[name] = h
    return out


def test_defaults_are_reference_of_record():
    assert fz.OUT_DIR == _dir("cifar10") == common.FROZEN_DIR / "cifar10_int8_r2"
    assert fz.CKPT_REL == _ckpt("cifar10") == "v2/model/retrain/cifar10_fp32_r2.pt"
    assert NET_CONFIGS["cifar10"]["quant_params"] == fz.OUT_DIR / fz.NPZ_NAME


def test_r2_checkpoint_sha256_file():
    line = (common.REPO_ROOT / (_ckpt("cifar10") + ".sha256")).read_text().split()
    assert line == [common.sha256_file(common.REPO_ROOT / _ckpt("cifar10")), "cifar10_fp32_r2.pt"]


@pytest.mark.parametrize("key", KEYS)
def test_sha256sums_cover_and_match(key):
    sums = _sums(key)
    assert set(sums) == {fz.NPZ_NAME, fz.MANIFEST_NAME}
    for name, h in sums.items():
        assert common.sha256_file(_dir(key) / name) == h, name


@pytest.mark.parametrize("key", KEYS)
def test_manifest_consistent(key):
    npz = _dir(key) / fz.NPZ_NAME
    m = json.loads((_dir(key) / fz.MANIFEST_NAME).read_text())
    assert m["outputs"][fz.NPZ_NAME] == common.sha256_file(npz)
    assert m["source_commit"] == fz.source_commit()
    assert m["checkpoint"] == {"path": _ckpt(key),
                               "sha256": common.sha256_file(common.REPO_ROOT / _ckpt(key))}
    for path, h in m["legacy_code"].items():
        assert common.sha256_file(common.REPO_ROOT / path) == h, path
    assert m["legacy_inputs_unchanged_since_source_commit"] is True
    with np.load(npz) as z:
        assert set(m["arrays"]) == set(z.files)
        for k in z.files:
            assert m["arrays"][k] == {"shape": list(z[k].shape), "dtype": str(z[k].dtype)}


@pytest.mark.parametrize("key", KEYS)
def test_keys_and_dtypes(key):
    with np.load(_dir(key) / fz.NPZ_NAME) as z:
        assert set(z.files) == fz.expected_keys()
        for name in fz.LAYERS:
            assert z[f"{name}_q_w"].dtype == np.int8
            assert z[f"{name}_q_b"].dtype == np.int32
            assert z[f"{name}_S_w"].dtype == np.float64
            assert z[f"{name}_S_a"].dtype == np.float64
            if name in fz.REQUANT_LAYERS:
                assert z[f"{name}_M"].dtype == np.float64
                assert z[f"{name}_S_out"].dtype == np.float64
            else:
                assert f"{name}_M" not in z.files and f"{name}_S_out" not in z.files
        assert z["S_input"].dtype == np.float64


def test_r1_r2_same_shapes_different_values():
    with np.load(_dir("cifar10") / fz.NPZ_NAME) as a, np.load(_dir("cifar10_r1") / fz.NPZ_NAME) as b:
        assert set(a.files) == set(b.files)
        for k in a.files:
            assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
        assert not np.array_equal(a["conv1_q_w"], b["conv1_q_w"])


@pytest.fixture(scope="module", params=KEYS)
def legacy(request):
    return request.param, fz.legacy_params(ckpt_rel=_ckpt(request.param))


def test_npz_equals_legacy_calibrate(legacy):
    key, params = legacy
    ref = fz.params_to_arrays(params)
    with np.load(_dir(key) / fz.NPZ_NAME) as z:
        assert set(z.files) == set(ref)
        for k in z.files:
            assert z[k].dtype == ref[k].dtype and np.array_equal(z[k], ref[k]), k


def test_roundtrip_bit_identical(legacy):
    key, params = legacy
    assert fz.check_roundtrip(params, _dir(key) / fz.NPZ_NAME) == len(fz.ROUNDTRIP_IDX)


def test_registered_dir_refuses_other_checkpoint():
    with pytest.raises(SystemExit, match="refusing"):
        fz.export(_dir("cifar10_r1"), _ckpt("cifar10"))
    with pytest.raises(SystemExit, match="refusing"):
        fz.export(_dir("cifar10"), _ckpt("cifar10_r1"))


@pytest.mark.slow
@pytest.mark.parametrize("key", KEYS)
def test_export_deterministic(tmp_path, key):
    """Two fresh-process exports reproduce the committed files byte-for-byte."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for i in (1, 2):
        out = tmp_path / f"run{i}"
        subprocess.run([sys.executable, str(common.MODEL_DIR / "freeze_cifar10_int8.py"),
                        "--ckpt", _ckpt(key), "--out", str(out)], check=True, env=env,
                       cwd=common.REPO_ROOT, capture_output=True)
        for name, h in _sums(key).items():
            assert common.sha256_file(out / name) == h, (i, name)


@pytest.mark.slow
@pytest.mark.parametrize("key", KEYS)
def test_frozen_npz_full_test_set_matches_record(key):
    """Full 10k: frozen npz and legacy calibrate give the same count as the CSV of record."""
    import reference_accuracy as ra
    c_leg, c_frz, _ = ra.cifar_int8_counts(key)
    assert c_frz == c_leg
    ver = NET_CONFIGS[key]["reference_version"]
    with open(ra.CSV_PATH) as f:
        rec = [r for r in csv.DictReader(f)
               if r["reference_version"] == ver and r["precision"] == "INT8"]
    assert len(rec) == 1 and int(rec[0]["correct"]) == c_frz
