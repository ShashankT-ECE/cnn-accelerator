"""Step 2.1b: retrained CIFAR-10 r2 artifacts (not the reference of record)."""
import json

import numpy as np
import pytest

import common
import freeze_cifar10_int8 as fz

R2_CKPT = "v2/model/retrain/cifar10_fp32_r2.pt"
R2_DIR = common.FROZEN_DIR / "cifar10_int8_r2"
R2_NPZ = R2_DIR / fz.NPZ_NAME

pytestmark = pytest.mark.skipif(not R2_NPZ.exists(), reason="r2 not frozen")


def test_checkpoint_sha256_file():
    line = (common.REPO_ROOT / (R2_CKPT + ".sha256")).read_text().split()
    assert line == [common.sha256_file(common.REPO_ROOT / R2_CKPT), "cifar10_fp32_r2.pt"]


def test_sha256sums_and_manifest():
    for line in (R2_DIR / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        assert common.sha256_file(R2_DIR / name) == h, name
    m = json.loads((R2_DIR / fz.MANIFEST_NAME).read_text())
    assert m["checkpoint"] == {"path": R2_CKPT,
                               "sha256": common.sha256_file(common.REPO_ROOT / R2_CKPT)}
    assert m["outputs"][fz.NPZ_NAME] == common.sha256_file(R2_NPZ)
    assert m["legacy_inputs_unchanged_since_source_commit"] is True


def test_npz_equals_legacy_calibrate_of_r2():
    params = fz.legacy_params(ckpt_rel=R2_CKPT)
    qp = fz.params_to_arrays(params)
    with np.load(R2_NPZ) as z:
        assert set(z.files) == set(qp) == fz.expected_keys()
        for k in z.files:
            assert z[k].dtype == qp[k].dtype and np.array_equal(z[k], qp[k]), k


def test_r2_differs_from_reference_of_record():
    with np.load(R2_NPZ) as a, np.load(fz.OUT_DIR / fz.NPZ_NAME) as b:
        assert not np.array_equal(a["conv1_q_w"], b["conv1_q_w"])


def test_same_shapes_as_reference_of_record():
    with np.load(R2_NPZ) as a, np.load(fz.OUT_DIR / fz.NPZ_NAME) as b:
        assert set(a.files) == set(b.files)
        for k in a.files:
            assert a[k].shape == b[k].shape and a[k].dtype == b[k].dtype, k
