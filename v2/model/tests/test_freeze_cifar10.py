"""Step 2.1: frozen CIFAR-10 INT8 reference (v2/model/frozen/cifar10_int8)."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import common
import freeze_cifar10_int8 as fz

FROZEN = fz.OUT_DIR
NPZ = FROZEN / fz.NPZ_NAME


def _sums():
    out = {}
    for line in (FROZEN / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        out[name] = h
    return out


def test_sha256sums_cover_and_match():
    sums = _sums()
    assert set(sums) == {fz.NPZ_NAME, fz.MANIFEST_NAME}
    for name, h in sums.items():
        assert common.sha256_file(FROZEN / name) == h, name


def test_manifest_consistent():
    m = json.loads((FROZEN / fz.MANIFEST_NAME).read_text())
    assert m["outputs"][fz.NPZ_NAME] == common.sha256_file(NPZ)
    assert m["source_commit"] == fz.source_commit()
    assert m["checkpoint"]["sha256"] == common.sha256_file(common.REPO_ROOT / fz.CKPT_REL)
    for path, h in m["legacy_code"].items():
        assert common.sha256_file(common.REPO_ROOT / path) == h, path
    assert m["legacy_inputs_unchanged_since_source_commit"] is True
    with np.load(NPZ) as z:
        assert set(m["arrays"]) == set(z.files)
        for k in z.files:
            assert m["arrays"][k] == {"shape": list(z[k].shape), "dtype": str(z[k].dtype)}


def test_keys_and_dtypes():
    with np.load(NPZ) as z:
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


@pytest.fixture(scope="module")
def legacy():
    return fz.legacy_params()


def test_npz_equals_legacy_calibrate(legacy):
    ref = fz.params_to_arrays(legacy)
    with np.load(NPZ) as z:
        assert set(z.files) == set(ref)
        for k in z.files:
            assert z[k].dtype == ref[k].dtype and np.array_equal(z[k], ref[k]), k


def test_roundtrip_bit_identical(legacy):
    assert fz.check_roundtrip(legacy, NPZ) == len(fz.ROUNDTRIP_IDX)


@pytest.mark.slow
def test_export_deterministic(tmp_path):
    """Two fresh-process exports reproduce the committed files byte-for-byte."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for i in (1, 2):
        out = tmp_path / f"run{i}"
        subprocess.run([sys.executable, str(common.MODEL_DIR / "freeze_cifar10_int8.py"),
                        "--out", str(out)], check=True, env=env, cwd=common.REPO_ROOT,
                       capture_output=True)
        for name, h in _sums().items():
            assert common.sha256_file(out / name) == h, (i, name)


@pytest.mark.slow
def test_frozen_npz_full_test_set_matches_record():
    """Full 10k: frozen npz and legacy calibrate give the same count as the CSV of record."""
    import reference_accuracy as ra
    c_leg, c_frz, _ = ra.cifar_int8_counts()
    assert c_frz == c_leg
    with open(ra.CSV_PATH) as f:
        rec = [r for r in csv.DictReader(f) if r["net"] == "cifar10" and r["precision"] == "INT8"]
    assert len(rec) == 1 and int(rec[0]["correct"]) == c_frz
