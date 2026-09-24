"""Step 2.1: LeNet INT8 reference is a pointer to the legacy frozen file (not a copy)."""
import common

LEGACY = common.REPO_ROOT / "data" / "lenet5_int8"
POINTER = common.FROZEN_DIR / "lenet5_int8" / "POINTER.md"


def _legacy_sum(name):
    for line in (LEGACY / "SHA256SUMS").read_text().splitlines():
        h, n = line.split("  ", 1)
        if n == name:
            return h
    raise KeyError(name)


def test_lenet_quant_params_sha256():
    assert common.sha256_file(LEGACY / "quant_params.npz") == _legacy_sum("quant_params.npz")


def test_pointer_names_file_and_hash():
    text = POINTER.read_text()
    assert "data/lenet5_int8/quant_params.npz" in text
    assert _legacy_sum("quant_params.npz") in text


def test_no_copy_under_v2():
    # Only the V2-derived hw_requant.npz may live here; no copy of the legacy params.
    d = common.FROZEN_DIR / "lenet5_int8"
    legacy = {common.sha256_file(p) for p in LEGACY.glob("*.npz")}
    for p in d.glob("*.npz"):
        assert p.name == "hw_requant.npz", p
        assert common.sha256_file(p) not in legacy, p
